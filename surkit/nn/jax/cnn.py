#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import Dict, Union, Tuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import activations, pooling
from .utils import Upsample, Flatten, Dropout2D, Dropout


def _pair(kernel_size):
    if isinstance(kernel_size, tuple):
        return kernel_size
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)

class Conv2d(nn.Module):
    """
    Conv Layer of BayesianNN.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (int | tuple[int, int]): size of the conv kernel
        stride (int | tuple[int, int], optional): stride of the convolution
        padding (int | tuple[int, int] | str, optional): padding added to all four sides of the input
        dilation (int | tuple[int, int], optional): spacing between kernel elements
        groups (int, optional): number of blocked connections from input channels to output channels
        bias (bool, optional): if True, adds a learnable bias to the output
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int], str] = 0
    dilation: Union[int, Tuple[int, int]] = 1
    groups: int = 1
    bias: bool = False

    def setup(self):
        self.w = self.param('w', jax.nn.initializers.glorot_normal(), (self.out_channels, self.in_channels, *_pair(self.kernel_size)))
        if self.bias:
            self.b = self.param('b', jax.nn.initializers.zeros, (self.out_channels))

    def __call__(self, x):
        x = jax.lax.conv_general_dilated(x, self.w, _pair(self.stride), ((_pair(self.padding)[0], _pair(self.padding)[1]),
                                                                         (_pair(self.padding)[0], _pair(self.padding)[1])),
                                         rhs_dilation=_pair(self.dilation), feature_group_count=self.groups)

        if self.bias:
            b = self.b.reshape(-1, 1, 1)
            x += b

        return x


class TransConv2d(nn.Module):
    """
    Transposed Conv Layer of BayesianNN.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (int | tuple[int, int]): size of the conv kernel
        stride (int | tuple[int, int], optional): stride of the convolution
        padding (int | tuple[int, int], optional): padding added to all four sides of the input
        output_padding (int | tuple[int, int], optional): padding added to all four sides of the output
        dilation (int | tuple[int, int], optional): spacing between kernel elements
        groups (int, optional): number of blocked connections from input channels to output channels
        bias (bool, optional): if True, adds a learnable bias to the output.
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int]] = 0
    output_padding: Union[int, Tuple[int, int]] = 0
    dilation: Union[int, Tuple[int, int]] = 1
    groups: int = 1
    bias: bool = False

    def setup(self):
        self.w = self.param('w', jax.nn.initializers.glorot_normal(),
                            (self.out_channels, self.in_channels, *_pair(self.kernel_size)))
        if self.bias:
            self.b = self.param('b', jax.nn.initializers.zeros, (self.out_channels))

    def __call__(self, x, train=False):
        _kernel_size = _pair(self.kernel_size)
        _dilation = _pair(self.dilation)
        _stride = _pair(self.stride)
        _padding = _pair(self.padding)
        _outpadding = _pair(self.output_padding)

        padding_before = lambda i : _kernel_size[i] - 1 + (_kernel_size[i] - 1) * (_dilation[i] - 1) - _padding[i]
        padding_after = lambda i : padding_before(i) + _outpadding[i]

        x = jax.lax.conv_transpose(x,
                                   self.w,
                                   padding=((padding_before(0), padding_after(0)),
                                            (padding_before(1), padding_after(1))),
                                   strides=(_stride[0], _stride[1]),
                                   rhs_dilation=(_dilation[0], _dilation[1]),
                                   dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                                   transpose_kernel=True)

        if self.bias:
            b = self.b.reshape(-1, 1, 1)
            x += b

        return x



class Cnn(nn.Module):
    """
    CNN model (image to vector) according to the given layers config

    Args:
        layers (list[dict[str, str | int]]): list with config of each layer
        img_size (list[int] | tuple[int, int] | int]): input size
        in_channel (int): input channels
        out_d (int): output length
        train (bool): training step
    """

    layers: Sequence[Dict[str, Union[str, int]]]
    img_size: Union[Sequence[int], Tuple[int, int], int]
    in_channel: int
    out_d: int
    train : bool = False

    def setup(self):
        if (isinstance(self.img_size, tuple) or isinstance(self.img_size, list)) and len(self.img_size) == 2:
            h = self.img_size[0]
            w = self.img_size[1]
        elif isinstance(self.img_size, int):
            h = self.img_size
            w = self.img_size
            self.img_size = (self.img_size, self.img_size)
        else:
            print(self.img_size)
            raise TypeError('img_size should be a tuple or list with length 2 or an int')

        in_channel = self.in_channel
        in_d = 0
        main = []
        for layer in self.layers:
            # conv layer
            if layer["type"] == 'conv':
                """
                可以添加默认参数
                i.e.
                size = (layer["size"], layer["size"]) if isinstance(layer["size"], int) else layer["size"]
                stride = (1, 1) if "stride" not in layer else layer["stride"]
                stride = (stride, stride) if isinstance(stride, int) else stride
                """

                main.append(Conv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                        padding=layer["padding"]))
                in_channel = layer["out_channel"]
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            elif layer["type"] == 'deconv' or layer["type"] == 'transconv':
                main.append(
                    TransConv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                     padding=layer["padding"], output_padding=layer["output_padding"]))
                in_channel = layer["out_channel"]
                h = (h - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]
                w = (w - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]

            # upsampling layer
            elif layer["type"] == 'upsample':
                # default mode is bilinear
                mode = layer["scale_factor"] if "scale_factor" in layer else 'bilinear'
                # jax不支持align_corners
                main.append(Upsample(scale_factor=layer["scale_factor"], mode=mode))
                # if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                #     main.append(Upsample(scale_factor=layer["scale_factor"], mode=mode))
                # else:
                #     main.append(Upsample(scale_factor=layer["scale_factor"], mode=mode))
                w *= layer["scale_factor"]
                h *= layer["scale_factor"]

            # pooling layer
            elif layer["type"] == 'pool' or layer["type"] == 'pooling':
                main.append(pooling.get(layer["name"])(layer["size"], stride=layer["stride"],
                                                       padding=layer["padding"]))
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            # batch normalization layer
            elif layer["type"] == 'bn':
                main.append(nn.BatchNorm(use_running_average=not self.train, momentum=0.1, epsilon=1e-5, axis=1,
                                         dtype=jnp.float32))
            # dropout
            elif layer["type"] == 'dropout':
                if in_channel > 1:
                    main.append(Dropout2D(self.p, broadcast_dims=[2, 3]))
                else:
                    main.append(Dropout(p=layer["rate"]))

            # activation layer
            elif layer["type"] == 'activation':
                main.append(activations.get(layer["name"]))

            # flatten layer
            elif layer["type"] == 'flatten':
                main.append(Flatten())
                in_d = int(w * h * in_channel)
                in_channel = 1

            # fully-connect layer
            elif layer["type"] == 'fc' or layer["type"] == 'dense':
                if in_d == 0:
                    raise TypeError('fully-connect layer should be applied after the flatten layer')
                main.append(nn.Dense(layer["out_d"]))
                in_d = layer["out_d"]

            # unexpected type
            else:
                raise ValueError("unexpected type %s" % (layer["type"]))

        # 改；不一定要全连接层
        main.append(nn.Dense(self.out_d))

        self.main = main

    def __call__(self, x, train=False):
        for layer in self.main:
            if train:
                # dropout
                try:
                    x = layer(x, train)
                # fc
                except:
                    x = layer(x)
            else:
                x = layer(x)
        return x
