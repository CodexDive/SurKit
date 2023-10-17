#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn

from . import initializers, activations
from .transform import transform


class ExcitationBlock(nn.Module):
    """
    One-dimensional excitation blocks，produces per-channel modulation weights.

    Args:
        size (int): the size of the weight parameter.
        initializer (callable, optional): the function to initialize the weight parameter.
    """
    def __init__(self, size, initializer=nn.init.xavier_normal_):
        super(ExcitationBlock, self).__init__()
        self.w = nn.Parameter(torch.zeros(1, size))
        initializer(self.w)

    def forward(self, x):
        return torch.mul(self.w, x)


class FNN(nn.Module):
    """
    Feedforward Neural Network.

    Args:
        layers (list[int]): list with numbers output feature at each layer
        activation (str | nn.Module): name of an activation function or a callable activation function
        in_d (int): input length
        out_d (int): output length
        initializer (str | callable, optional): name of an initializer or a callable initializer function
        transforms (list[str], optional): perform affine transformations on the input
        excitation (str, optional): mode of excitation, "fixed" or "unfixed" or None
    """
    def __init__(
            self,
            layers,
            activation,
            in_d,
            out_d,
            initializer="He uniform",
            transforms=None,
            excitation=None,
    ):
        super(FNN, self).__init__()
        self.main = nn.ModuleList([])
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.transforms = transforms
        self.excitation = excitation
        if transforms:
            in_d *= len(transforms) + 1
        if excitation == 'fixed':
            if layers.count(layers[0]) == len(layers):
                self.excitation_block = ExcitationBlock(layers[0], self.initializer)
            else:
                raise RuntimeError("for fixed excitation, number of neurons in layers should be same")
        for layer in layers:
            self.main.append(nn.Linear(in_d, layer))
            self.initializer(self.main[-1].weight)
            nn.init.uniform_(self.main[-1].bias)
            self.main.append(self.activation)
            if excitation == 'fixed':
                self.main.append(self.excitation_block)
            elif excitation == 'unfixed':
                self.main.append(ExcitationBlock(layer, self.initializer))
            in_d = layer
        self.main.append(nn.Linear(in_d, out_d))

    # This function defines the forward rule of output respect to input.
    def forward(self, x):
        if self.transforms:
            xs = [x]
            for t in self.transforms:
                xs.append(transform(t, x))
            x = torch.concat(xs, 1)
        for layer in self.main:
            x = layer(x)
        return x

class PFNN(nn.Module):
    """
    Parallel FNN, each output feature is fitted with a separate FNN.

    Args:
        layers (list[int]): list with numbers output feature at each layer
        activation (str | nn.Module): name of an activation function or a callable activation function
        in_d (int): input length
        out_d (int): output length
        initializer (str | callable, optional): name of an initializer or a callable initializer function
        transforms (list[str], optional): perform affine transformations on the input
        excitation (str, optional): mode of excitation, "fixed" or "unfixed" or None
    """
    def __init__(
            self,
            layers,
            activation,
            in_d=1,
            out_d=1,
            initializer="He uniform",
            transforms=None,
            excitation=None,
    ):

        super(PFNN, self).__init__()
        self.main = nn.ModuleList([])
        for i in range(out_d):
            self.main.append(FNN(layers, activation, in_d, 1, initializer, transforms, excitation))

    # This function defines the forward rule of output respect to input.
    def forward(self, x):
        y = []
        for n in self.main:
            y.append(n(x))
        x = torch.cat(y, dim=1)
        return x
