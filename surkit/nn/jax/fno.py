#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import jax
import jax.numpy as jnp
import flax.linen as nn

from . import initializers, activations

from .fno_conv import SpectralConv2d
from .cnn import Conv2d


class FNO2d(nn.Module):
    modes1: int
    modes2: int
    width: int = 64
    in_channels: int = 3
    out_channels: int = 1
    activation: str = 'gelu'

    def setup(self):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.padding = 2  # pad the domain if input is non-periodic

        self.fc0 = nn.Dense(self.width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = Conv2d(self.width, self.width, 1)
        self.w1 = Conv2d(self.width, self.width, 1)
        self.w2 = Conv2d(self.width, self.width, 1)
        self.w3 = Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(self.out_channels)

        self.act = activations.get(self.activation)

    def __call__(self, x, train=False):
        x = self.fc0(x)
        x = x.transpose(0, 3, 1, 2)
        x = jnp.pad(x, [(0, 0), (0, 0), (0, self.padding), (0, self.padding)], mode='constant')

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.transpose(0, 2, 3, 1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x