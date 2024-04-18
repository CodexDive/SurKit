#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import jax
import jax.numpy as jnp
from flax import linen as nn



class SpectralConv2d(nn.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def setup(self):
        def complex_uniform(key, shape):
            return self.scale * jnp.array(jax.random.uniform(key, shape), dtype=jnp.complex64)
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = self.param('weights1', complex_uniform, (self.in_channels, self.out_channels, self.modes1, self.modes2))
        self.weights2 = self.param('weights2', complex_uniform, (self.in_channels, self.out_channels, self.modes1, self.modes2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y ), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return jnp.einsum("bixy,ioxy->boxy", input, weights)

    def __call__(self, x):
        batchsize = x.shape[0]
        x_ft = jnp.fft.rfft2(x)
        out_ft = jnp.zeros([batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1], dtype=jnp.complex64)
        out_ft.at[:, :, :self.modes1, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        )
        out_ft.at[:, :, -self.modes1:, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        )
        x = jnp.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x