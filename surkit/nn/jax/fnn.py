#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from typing import Sequence

import jax.numpy
from flax import linen as nn
from flax.linen import compact

from . import initializers, activations
from .transform import transform


class ExcitationBlock(nn.Module):
    """
    One-dimensional excitation blocks，produces per-channel modulation weights.

    Args:
        size (int): the size of the weight parameter.
        initializer (callable, optional): the function to initialize the weight parameter.
    """

    size: int
    initializer: nn.initializers.Initializer = jax.nn.initializers.glorot_normal

    @compact
    def call(self, x):
        w = self.param('excitation_block', self.initializer, (1, self.size))
        return w * x



class FNN(nn.Module):
    """
    Feedforward Neural Network.

    Args:
        layers (list[int]): list with numbers output feature at each layer
        activation (str): name of an activation function or a callable activation function
        in_d (int): input length
        out_d (int): output length
        initializer (str | callable, optional): name of an initializer or a callable initializer function
        transforms (list[str], optional): perform affine transformations on the input
        excitation (str, optional): mode of excitation, "fixed" or "unfixed" or None
    """

    layers: Sequence[int]
    activation: str
    in_d: int
    out_d: int
    initializer: str = "He uniform"
    transforms: Sequence[str] = None #: Sequence[str]
    excitation: str = None

    def setup(self):
        main = []
        init_fn = initializers.get(self.initializer)
        act_func = activations.get(self.activation)
        excitation_block = None
        if self.excitation == 'fixed':
            if self.layers.count(self.layers[0]) == len(self.layers):
                excitation_block = self.param('excitation_block', init_fn, (1, self.layers[0]))
            else:
                raise RuntimeError("for fixed excitation, number of neurons in layers should be same")
        count = 0
        for i in self.layers:
            main.append(nn.Dense(features=i, kernel_init=init_fn))
            main.append(act_func)
            if self.excitation == 'fixed':
                main.append(excitation_block)
            if self.excitation == 'unfixed':
                main.append(self.param('excitation_block_{}'.format(count), init_fn, (1, i)))
                count += 1
        main.append(nn.Dense(features=self.out_d))
        self.main = main


    def __call__(self, x):
        if self.transforms:
            xs = [x]
            for t in self.transforms:
                xs.append(transform(t, x))
            x = jax.numpy.concatenate(xs, axis=1)
        for layer in self.main[:-1]:
            try:
                x = layer(x)
            except:
                x = layer * x


        return self.main[-1](x)


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

    layers: Sequence[int]
    activation: str
    in_d: int
    out_d: int
    initializer: str = "He uniform"
    transforms: Sequence[str] = None  #: Sequence[str]
    excitation: str = None


    def setup(self):
        main = []
        for i in range(self.out_d):
            main.append(FNN(self.layers, self.activation, self.in_d, 1, self.initializer, self.transforms, self.excitation))
        self.main = main

    def __call__(self, x):
        y = []
        for n in self.main:
            y.append(n(x))
        x = jax.numpy.concatenate(y, axis=1)
        return x