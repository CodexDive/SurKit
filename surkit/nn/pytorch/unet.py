#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
from . import activations

class block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation='tanh'):
        super().__init__()
        self.act = activations.get(activation)
        if not mid_channels: mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1 , bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1 , bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, first_features=32, activation='tanh'):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_conv = block(in_channels, first_features, activation=activation)

        self.encoder1 = block(first_features, 2 * first_features, activation=activation)
        self.encoder2 = block(2 * first_features, 4 * first_features, activation=activation)
        self.encoder3 = block(4 * first_features, 8 * first_features, activation=activation)
        self.encoder4 = block(8 * first_features, 16 * first_features, activation=activation)


        self.up1 = nn.ConvTranspose2d(16 * first_features, 8 * first_features, 2, 2)
        self.decoder1 = block(16 * first_features, 8 * first_features, activation=activation)
        self.up2 = nn.ConvTranspose2d(8 * first_features, 4 * first_features, 2, 2)
        self.decoder2 = block(8 * first_features, 4 * first_features, activation=activation)
        self.up3 = nn.ConvTranspose2d(4 * first_features, 2 * first_features, 2, 2)
        self.decoder3 = block(4 * first_features, 2 * first_features, activation=activation)
        self.up4 = nn.ConvTranspose2d(2 * first_features, first_features, 2, 2)
        self.decoder4 = block(2 * first_features, first_features, activation=activation)

        self.out_conv = nn.Conv2d(first_features, out_channels, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encoder1(self.pool(x1))
        x3 = self.encoder2(self.pool(x2))
        x4 = self.encoder3(self.pool(x3))

        x = self.encoder4(self.pool(x4))

        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.up2(self.decoder1(x))
        x = torch.cat((x, x3), dim=1)
        x = self.up3(self.decoder2(x))
        x = torch.cat((x, x2), dim=1)
        x = self.up4(self.decoder3(x))
        x = torch.cat((x, x1), dim=1)
        x = self.out_conv(self.decoder4(x))
        return x
