#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
import oneflow.nn as nn

class Swish(nn.Module):
    """
    A smooth, non-monotonic activation function.

    Swish is defined as:

    .. math::
        \\text{Swish}(x) = x \\times \\text{Sigmoid}(x)
    """
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(flow.sigmoid(x))
            return x
        else:
            return x * flow.sigmoid(x)


def get(activation):
    """
    Return an activation function based on the given string, or return the given callable activation function
    Args:
        activation (str | nn.Module): name of an activation function or a callable oneflow activation function

    Returns:
        nn.Module: a callable activation function

    """
    activation_dict = {
        "swish": Swish(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
    }

    if isinstance(activation, str):
        return activation_dict[activation.lower()]

    if callable(activation):
        return activation
