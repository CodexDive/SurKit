#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

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

    layers: Sequence[int]
    activation: str
    in_d: int
    out_d: int
    transforms: Sequence[str] = None  #: Sequence[str]
    noise_tol: float = .1
    prior_mean: float = 0.
    prior_var: float = 0.1

    def setup(self):
        main = []
        in_d = self.in_d
        if self.transforms:
            in_d *= len(self.transforms) + 1
        for layer in self.layers:
            main.append(BayesLinear(in_d, layer, prior_mean=self.prior_mean, prior_var=self.prior_var))
            in_d = layer
        main.append(BayesLinear(in_d, self.out_d))
        self.main = main
        self.act_func = activations.get(self.activation)

    def __call__(self, x, train=False):
        priors = 0.
        posts = 0.
        if self.transforms:
            xs = [x]
            for t in self.transforms:
                xs.append(transform(t, x))
            x = jnp.concatenate(xs, 1)
        for layer in self.main[:-1]:
            if train:
                x, prior, post = layer(x, train)
                priors += prior
                posts += post
            else:
                x = layer(x)
            x = self.act_func(x)
        if train:
            x, prior, post = self.main[-1](x, train)
            priors += prior
            posts += post
            return x, priors, posts
        x = self.main[-1](x)
        return x