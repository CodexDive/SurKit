#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
import oneflow.nn.functional as F
from oneflow import nn

from .fnn import FNN


class GaussianNN(FNN):
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
        super(GaussianNN, self).__init__(layers, activation, in_d, out_d+1, initializer, transforms, excitation)
        self.out_d = out_d + 1

    def forward(self, x):
        x = super(GaussianNN, self).forward(x)
        mean, variance = flow.split(x, self.out_d-1, dim=1)
        variance = F.softplus(variance) + 1e-6
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
    def __init__(
            self,
            n_models,
            layers,
            activation,
            in_d,
            out_d,
            initializer="He normal",
            transforms=None,
            excitation=None,
    ):
        super(MixtureNN, self).__init__()
        self.n_models = n_models
        self.models = nn.ModuleList()
        for i in range(n_models):
            self.models.append(GaussianNN(layers, activation, in_d, out_d, initializer, transforms, excitation))

    def forward(self, x):
        means = []
        variances = []
        for model in self.models:
            mean, variance = model(x)
            means.append(mean)
            variances.append(variance)
        means = flow.stack(means)
        mean = means.mean(dim=0)
        variances = flow.stack(variances)
        variance = (variances + means.pow(2)).mean(0) - mean.pow(2)
        # variance = F.softplus(variance) + 1e-6
        return mean, variance
