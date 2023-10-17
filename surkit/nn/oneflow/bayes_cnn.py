#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import List, Dict, Union, Tuple

import oneflow as flow
from oneflow import nn

from . import activations, pooling
from .bayes_layer import BayesLinear, BayesConv2d, BayesTransConv2d


class BayesCnn(nn.Module):
    """
    BayesCnn model (image to vector) according to the given layers config

    Args:
        layers (list[dict[str, str | int]]): list with config of each layer
        img_size (list[int] | tuple[int, int] | int]): input size
        in_channel (int): input channels
        out_d (int): output length
        noise_tol (float, optional): noise tolerance, for calculate likelihood in negative elbo
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """
    def __init__(
            self,
            layers: List[Dict[str, Union[str, int]]],
            img_size: Union[List[int], Tuple[int, int], int],
            in_channel:int,
            out_d:int,
            noise_tol:float=.1,
            prior_mean:float=0.,
            prior_var:float=1.,
    ):

        super(BayesCnn, self).__init__()
        # avoid RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
        flow.backends.cudnn.enabled = False

        self.noise_tol = noise_tol
        # 改；可以用torch的_tuple()
        if (isinstance(img_size, tuple) or  isinstance(img_size, list)) and len(img_size) == 2:
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
            # conv layer
            if layer["type"] == 'conv':
                """
                可以添加默认参数
                i.e.
                size = (layer["size"], layer["size"]) if isinstance(layer["size"], int) else layer["size"]
                stride = (1, 1) if "stride" not in layer else layer["stride"]
                stride = (stride, stride) if isinstance(stride, int) else stride
                """

                self.main.append(BayesConv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                             padding=layer["padding"], prior_mean=prior_mean, prior_var=prior_var))
                in_channel = layer["out_channel"]
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            elif layer["type"] == 'deconv' or layer["type"] == 'transconv':
                self.main.append(
                    BayesTransConv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                     padding=layer["padding"], output_padding=layer["output_padding"],
                                     prior_mean=prior_mean, prior_var=prior_var))
                in_channel = layer["out_channel"]
                h = (h - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]
                w = (w - 1) * layer["stride"] + layer["size"] - 2 * layer["padding"] + 2 * layer["output_padding"]

            # upsampling layer
            elif layer["type"] == 'upsample':
                # default mode is bilinear
                mode = layer["scale_factor"] if "scale_factor" in layer else 'bilinear'
                self.main.append(nn.Upsample(scale_factor=layer["scale_factor"], mode=mode, align_corners=True))
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
                self.main.append(BayesLinear(in_d, layer["out_d"], prior_mean=prior_mean, prior_var=prior_var))
                in_d = layer["out_d"]

            # unexpected type
            else:
                raise ValueError("unexpected type %s" % (layer["type"]))

        # 改；不一定要全连接层
        self.main.append(BayesLinear(in_d, out_d, prior_mean=prior_mean, prior_var=prior_var))

        self.noise_tol = noise_tol

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        for layer in self.main:
            x = layer(x)
        return x

    def log_prior(self):
        """
        Calculate the log prior over all the layers
        """
        return sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self.main])

    def log_post(self):
        """
        Calculate the log posterior over all the layers
        """
        return sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self.main])
