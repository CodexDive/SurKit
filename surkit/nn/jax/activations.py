#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from flax import linen as nn

class Swish(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.swish(x)

class Relu(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.relu(x)

class LeakyRelu(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.leaky_relu(x)

class Tanh(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.tanh(x)

class Sigmoid(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.sigmoid(x)

class Softmax(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.softmax(x)

class SELU(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return nn.selu(x)

def get(activation):
    """
    Return an activation function based on the given string, or return the given callable activation function
    Args:
        activation (str | nn.Module): name of an activation function or a callable jax activation function

    Returns:
        nn.Module: a callable activation function

    """

    activation_dict = {
        "swish": Swish(),
        "relu": Relu(),
        "leaky_relu": LeakyRelu(),
        "tanh": Tanh(),
        "sigmoid": Sigmoid(),
        "softmax": Softmax(),
        "selu": SELU(),
    }

    if isinstance(activation, str):
        return activation_dict[activation.lower()]

    if callable(activation):
        return activation