#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import initializers, activations
from .transform import transform

from .fno_conv import SpectralConv1d, SpectralConv2d


class FNO1d(nn.Module):
    def __init__(
            self,
            modes,
            width,
            in_d = 2,
            out_d = 1,
            activation='gelu',
    ):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width

        self.fc0 = nn.Linear(in_d, self.width)  # input channel is 2: (a(x), x grid)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_d)

        self.activation = activations.get(activation)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 1, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FNO2d(nn.Module):
    def __init__(
            self,
            modes1,
            modes2,
            width=64,
            in_channels=3,
            out_channels=1,
            activation='gelu',
    ):
        super(FNO2d, self).__init__()

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

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.activation = activations.get(activation)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x