#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
from oneflow import nn

from . import activations
from .bayes_layer import BayesLinear
from .transform import transform


class BayesNN(nn.Module):
    """
    Bayesian FNN according to the given layers config

    Args:
        layers (list[int]): list with numbers output feature at each layer
        activation (str | nn.Module): name of an activation function or a callable activation function
        in_d (int): input length
        out_d (int): output length
        transforms (list[str], optional): perform affine transformations on the input
        noise_tol (float, optional): noise tolerance, for calculate likelihood in negative elbo
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """
    def __init__(
            self,
            layers,
            activation,
            in_d,
            out_d,
            transforms=None,
            noise_tol=.1,
            prior_mean=0.,
            prior_var=1.,
    ):
        super(BayesNN, self).__init__()
        self.main = nn.ModuleList([])
        self.activation = activations.get(activation)
        self.transforms = transforms
        self.noise_tol = noise_tol
        # self.excitation = excitation
        if transforms:
            in_d *= len(transforms) + 1

        for layer in layers:
            self.main.append(BayesLinear(in_d, layer, prior_mean=prior_mean, prior_var=prior_var))
            self.main.append(self.activation)
            in_d = layer
        self.main.append(BayesLinear(in_d, out_d))

    # This function defines the forward rule of output respect to input.
    def forward(self, x):
        if self.transforms:
            xs = [x]
            for t in self.transforms:
                xs.append(transform(t, x))
            x = flow.concat(xs, 1)
        for layer in self.main:
            x = layer(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self.main])

    def log_post(self):
        # calculate the log posterior over all the layers
        return sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self.main])
