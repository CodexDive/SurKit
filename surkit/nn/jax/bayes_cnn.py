#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import Dict, Union, Tuple, Sequence

import flax.linen as nn
import jax.numpy as jnp

from . import activations, pooling
from .bayes_layer import BayesLinear, BayesConv2d, BayesTransConv2d
from .utils import Upsample, Flatten, Dropout2D, Dropout


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
    layers: Sequence[Dict[str, Union[str, int]]]
    img_size: Union[Sequence[int], Tuple[int, int], int]
    in_channel: int
    out_d: int
    noise_tol: float = .1
    prior_mean: float = 0.
    prior_var: float = 1.
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

                main.append(BayesConv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                             padding=layer["padding"], prior_mean=self.prior_mean, prior_var=self.prior_var))
                in_channel = layer["out_channel"]
                h = (h + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1
                w = (w + 2 * layer["padding"] - layer["size"]) / layer["stride"] + 1

            elif layer["type"] == 'deconv' or layer["type"] == 'transconv':
                main.append(
                    BayesTransConv2d(in_channel, layer["out_channel"], layer["size"], stride=layer["stride"],
                                     padding=layer["padding"], output_padding=layer["output_padding"],
                                     prior_mean=self.prior_mean, prior_var=self.prior_var))
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
                main.append(BayesLinear(in_d, layer["out_d"], prior_mean=self.prior_mean, prior_var=self.prior_var))
                in_d = layer["out_d"]

            # unexpected type
            else:
                raise ValueError("unexpected type %s" % (layer["type"]))

        # 改；不一定要全连接层
        main.append(BayesLinear(in_d, self.out_d, prior_mean=self.prior_mean, prior_var=self.prior_var))

        self.main = main


    def __call__(self, x, train=False):
        priors = 0.
        posts = 0.
        for layer in self.main:
            if train:
                try:
                    # bayeslayer
                    x, prior, post = layer(x, train)
                    priors += prior
                    posts += post
                except:
                    # dropout
                    try: x = layer(x, train)
                    # fc
                    except:
                        x = layer(x)
            else:
                x = layer(x)
        if train:
            return x, priors, posts
        return x



