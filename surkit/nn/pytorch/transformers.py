#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert (
                hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor])
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor])
        o2_real = torch.zeros(x.shape)
        o2_imag = torch.zeros(x.shape)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        s = total_modes + kept_modes
        d = total_modes - kept_modes
        o1_real[:, d:s, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].real, self.w1[0]) -
            torch.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, d:s, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].imag, self.w1[0]) +
            torch.einsum("...bi,bio->...bo", x[:, d:s, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, d:s, :kept_modes] = (
                torch.einsum("...bi,bio->...bo", o1_real[:, d:s, :kept_modes], self.w2[0]) -
                torch.einsum("...bi,bio->...bo", o1_imag[:, d:s, :kept_modes], self.w2[1]) +
                self.b2[0]
        )

        o2_imag[:, d:s, :kept_modes] = (
                torch.einsum("...bi,bio->...bo", o1_imag[:, d:s, :kept_modes], self.w2[0]) +
                torch.einsum("...bi,bio->...bo", o1_real[:, d:s, :kept_modes], self.w2[1]) +
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias