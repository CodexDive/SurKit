#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import math
import flax.linen as nn
import jax
import jax.numpy as jnp

def softshrink(x, lambd=0.5):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lambd, 0.0)

def view_as_complex(x):
    return jnp.array(x[..., 0] + 1j * x[..., 1])



class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    hidden_size: int
    num_blocks: int = 8
    sparsity_threshold: float = 0.01
    hard_thresholding_fraction: int = 1
    hidden_size_factor: int = 1

    def setup(self):
        assert (
                self.hidden_size % self.num_blocks == 0
        ), f"hidden_size {self.hidden_size} should be divisble by num_blocks {self.num_blocks}"

        def randn(key, shape, dtype = float, minval = 0, maxval = 1):
            return self.scale * jax.random.uniform(key, shape, dtype, minval, maxval)
        self.block_size = self.hidden_size // self.num_blocks
        self.scale = 0.02

        self.w1 = self.param('weights1', randn, (2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = self.param('bias1', randn, (2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = self.param('weights2', randn, (2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = self.param('bias2', randn, (2, self.num_blocks, self.block_size))

    def __call__(self, x):
        bias = x

        dtype = x.dtype
        x = jnp.asarray(x, dtype=jnp.float32)
        B, H, W, C = x.shape
        x = jnp.fft.rfft2(x, axes=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = jnp.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor])
        o1_imag = jnp.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor])
        o2_real = jnp.zeros(x.shape)
        o2_imag = jnp.zeros(x.shape)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        s = total_modes + kept_modes
        d = total_modes - kept_modes
        o1_real = o1_real.at[:, d:s, :kept_modes].set(nn.relu(
            jnp.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].real, self.w1[0]) -
            jnp.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        ))

        o1_imag = o1_imag.at[:, d:s, :kept_modes].set(nn.relu(
            jnp.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].imag, self.w1[0]) +
            jnp.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        ))

        o2_real = o2_real.at[:, d:s, :kept_modes].set((
                jnp.einsum("...bi,bio->...bo", o1_real[:, d:s, :kept_modes], self.w2[0]) -
                jnp.einsum("...bi,bio->...bo", o1_imag[:, d:s, :kept_modes], self.w2[1]) +
                self.b2[0]
        ))

        o2_imag = o2_imag.at[:, d:s, :kept_modes].set((
                jnp.einsum("...bi,bio->...bo", o1_imag[:, d:s, :kept_modes], self.w2[0]) +
                jnp.einsum("...bi,bio->...bo", o1_real[:, d:s, :kept_modes], self.w2[1]) +
                self.b2[1]
        ))

        x = jnp.stack([o2_real, o2_imag], axis=-1)
        x = softshrink(x, lambd=self.sparsity_threshold)
        x = view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = jnp.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")
        x = jnp.asarray(x, dtype=dtype)

        return x + bias