#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import compact

from.cnn import Conv2d, TransConv2d
from.pooling import MaxPool2d
from . import activations

class block(nn.Module):
    in_channels: int
    out_channels: int
    mid_channels: int = None
    activation:str = 'tanh'
    def setup(self):
        self.act = activations.get(self.activation)
        if not self.mid_channels: mid_channels = self.out_channels
        else: mid_channels = self.mid_channels
        self.conv1 = Conv2d(self.in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv2 = Conv2d(mid_channels, self.out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm(momentum = 0.9)
        self.norm2 = nn.BatchNorm(momentum = 0.9)

    def __call__(self, x, train=False):
        x = self.conv1(x)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.norm1(x, use_running_average=not train)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = self.act(x)
        x = self.conv2(x)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.norm2(x, use_running_average=not train)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = self.act(x)
        return x

class UNet(nn.Module):
    in_channels: int = 3
    out_channels: int = 1
    first_features: int = 32
    mid_channels: int = None
    activation: str = 'tanh'
    def setup(self):
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.in_conv = block(self.in_channels, self.first_features, activation=self.activation)

        self.encoder1 = block(self.first_features, 2 * self.first_features, activation=self.activation)
        self.encoder2 = block(2 * self.first_features, 4 * self.first_features, activation=self.activation)
        self.encoder3 = block(4 * self.first_features, 8 * self.first_features, activation=self.activation)
        self.encoder4 = block(8 * self.first_features, 16 * self.first_features, activation=self.activation)

        self.up1 = TransConv2d(16 * self.first_features, 8 * self.first_features, 2, 2)
        self.decoder1 = block(16 * self.first_features, 8 * self.first_features, activation=self.activation)
        self.up2 = TransConv2d(8 * self.first_features, 4 * self.first_features, 2, 2)
        self.decoder2 = block(8 * self.first_features, 4 * self.first_features, activation=self.activation)
        self.up3 = TransConv2d(4 * self.first_features, 2 * self.first_features, 2, 2)
        self.decoder3 = block(4 * self.first_features, 2 * self.first_features, activation=self.activation)
        self.up4 = TransConv2d(2 * self.first_features, self.first_features, 2, 2)
        self.decoder4 = block(2 * self.first_features, self.first_features, activation=self.activation)

        self.out_conv = Conv2d(self.first_features, self.out_channels, 1)

    def __call__(self, x, train=False):
        x1 = self.in_conv(x, train)
        x2 = self.encoder1(self.pool(x1), train)
        x3 = self.encoder2(self.pool(x2), train)
        x4 = self.encoder3(self.pool(x3), train)

        x = self.encoder4(self.pool(x4), train)

        x = self.up1(x)
        x = jnp.concatenate((x, x4), 1)
        x = self.up2(self.decoder1(x, train))
        x = jnp.concatenate((x, x3), 1)
        x = self.up3(self.decoder2(x, train))
        x = jnp.concatenate((x, x2), 1)
        x = self.up4(self.decoder3(x, train))
        x = jnp.concatenate((x, x1), 1)
        x = self.out_conv(self.decoder4(x, train))
        return x
