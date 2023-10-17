#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from .fnn import FNN


class GaussianNN(nn.Module):
    """
    GaussianNN, outputs Gaussian distribution (mean and variance).

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
        main = FNN(self.layers, self.activation, self.in_d, self.out_d+1, self.initializer,  self.transforms, self.excitation)
        self.main = main

    def __call__(self, x):
        x = self.main(x)
        mean, variance = jnp.split(x, [self.out_d], axis=1)
        variance = nn.softplus(variance) + 1e-6  # Variance must be positive
        return mean, variance


class MixtureNN(nn.Module):
    """
    Bagging ensemble of GaussianNN.

    Args:
        n_models(int): number of networks participating in ensemble
        layers (list[int]): list with numbers output feature at each layer
        activation (str | nn.Module): name of an activation function or a callable activation function
        in_d (int): input length
        out_d (int): output length
        initializer (str | callable, optional): name of an initializer or a callable initializer function
        transforms (list[str], optional): perform affine transformations on the input
        excitation (str, optional): mode of excitation, "fixed" or "unfixed" or None
    """
    n_models: int
    layers: Sequence[int]
    activation: str
    in_d: int
    out_d: int
    initializer: str = "He uniform"
    transforms: Sequence[str] = None  #: Sequence[str]
    excitation: str = None

    def setup(self):
        main = []
        for i in range(self.n_models):
            main.append(GaussianNN(self.layers, self.activation, self.in_d, self.out_d, self.initializer,  self.transforms, self.excitation))
        self.main = main

    def __call__(self, x, train=False):
        means = []
        variances = []
        for n in self.main:
            mean, variance = n(x)
            means.append(mean)
            variances.append(variance)

        if train:
            return means, variances

        means = jnp.array(means)
        mean = means.mean(axis=0)
        variances = jnp.array(variances)
        variance = (variances + jnp.power(means, 2)).mean(axis=0) - jnp.power(mean, 2)
        variance = nn.softplus(variance) + 1e-6  # Variance must be positive
        return mean, variance