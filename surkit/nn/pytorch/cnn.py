#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import Union, Tuple, List

from torch import nn

from . import activations, pooling


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
    ):
        super(ConvBlock, self).__init__()
        self.main = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        return self.main(x)

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# class ResBlock(nn.Module):
#

class Cnn(nn.Module):
    """
    CNN model (image to vector) according to the given layers config

    Args:
        layers (list[dict[str, str | int]]): list with config of each layer
        img_size (list[int] | tuple[int, int] | int]): input size
        in_channel (int): input channels
        out_d (int): output length
    """
    def __init__(
            self,
            layers: List[dict],
            img_size: Union[int, Tuple[int, int]],
            in_channel: int,
            out_d: int,
    ):
        # initialize the nn like you would with a standard multilayer perceptron, but using the BBB layer
        super(Cnn, self).__init__()

        if (isinstance(img_size, tuple) or isinstance(img_size, list)) and len(img_size) == 2:
            self.img_size = img_size
            h = img_size[0]
            w = img_size[1]
        elif isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            h = img_size
            w = img_size
        else:
            print(img_size)
            raise TypeError('img_size should be a tuple or list with length 2 or an int')

        in_d = 0
        self.main = nn.ModuleList([])
        for layer in layers:
            # Conv layer
            if layer["type"] == 'conv':
                self.main.append(nn.Conv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                           padding=layer["padding"]))
                in_channel = layer["out_channel"]
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            elif layer["type"] == 'deconv' or layer["type"] == 'transconv':
                self.main.append(
                    nn.ConvTranspose2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                       padding=layer["padding"], output_padding=layer["output_padding"]))
                in_channel = layer["out_channel"]
                h = (h - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]
                w = (w - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]

            # upsampling layer
            elif layer["type"] == 'upsample':
                # default mode is bilinear
                mode = layer["scale_factor"] if "scale_factor" in layer else 'bilinear'
                if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                    self.main.append(nn.Upsample(scale_factor=layer["scale_factor"], mode=mode, align_corners=True))
                else:
                    self.main.append(nn.Upsample(scale_factor=layer["scale_factor"], mode=mode))
                w *= layer["scale_factor"]
                h *= layer["scale_factor"]

            # pooling layer
            elif layer["type"] == 'pool' or layer["type"] == 'pooling':
                self.main.append(pooling.get(layer["name"])(layer["size"], stride=layer["stride"],
                                                            padding=layer["padding"]))
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            # batch normalization layer
            elif layer["type"] == 'bn':
                self.main.append(nn.BatchNorm2d(in_channel))

            # dropout
            elif layer["type"] == 'dropout':
                self.main.append(nn.Dropout2d(p=layer["rate"]))

            # activation layer
            elif layer["type"] == 'activation':
                self.main.append(activations.get(layer["name"]))

            # flatten layer
            elif layer["type"] == 'flatten':
                self.main.append(nn.Flatten())
                in_d = int(w * h * in_channel)

            # fully-connect layer
            elif layer["type"] == 'fc' or layer["type"] == 'dense':
                self.main.append(nn.Linear(in_d, layer["out_d"]))
                in_d = layer["out_d"]

            # unexpected type
            else:
                raise ValueError("unexpected type %s" % (layer["type"]))

        self.main.append(nn.Linear(in_d, out_d))

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        for layer in self.main:
            x = layer(x)
        return x
