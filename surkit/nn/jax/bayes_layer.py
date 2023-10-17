#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from typing import Union, Tuple

import jax.lax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from flax import linen as nn


class BayesLinear(nn.Module):
    """
    Linear Layer of BayesianNN.

    Args:
        in_d (int): input length
        out_d (int): output length
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """
    in_d: int
    out_d: int
    prior_mean: float = 0.0
    prior_var: float = 1.0

    def setup(self):
        self.w_mu = self.param('w_mu', nn.initializers.zeros, (self.out_d, self.in_d))
        self.w_rho = self.param('w_rho', nn.initializers.zeros, (self.out_d, self.in_d))
        self.b_mu = self.param('b_mu', nn.initializers.zeros, (self.out_d))
        self.b_rho = self.param('b_rho', nn.initializers.zeros, (self.out_d))

    def __call__(self, x, train=False):
        # w_epsilon = torch.distributions.Normal(0, 1).sample(self.w_mu.shape).numpy()

        w_epsilon = np.random.normal(0, 1, self.w_mu.shape)
        w = self.w_mu + jnp.log(1 + jnp.exp(self.w_rho)) * w_epsilon

        b_epsilon = np.random.normal(0, 1, self.b_mu.shape)
        # b_epsilon = torch.distributions.Normal(0, 1).sample(self.b_mu.shape).numpy()
        b = self.b_mu + jnp.log(1 + jnp.exp(self.b_rho)) * b_epsilon

        w_log_prior = stats.norm.logpdf(w, self.prior_mean, jnp.sqrt(self.prior_var))
        b_log_prior = stats.norm.logpdf(b, self.prior_mean, jnp.sqrt(self.prior_var))

        log_prior = jnp.sum(w_log_prior) + jnp.sum(b_log_prior)
        w_log_post = stats.norm.logpdf(w, self.w_mu, jnp.log(1 + jnp.exp(self.w_rho)))
        b_log_post = stats.norm.logpdf(b, self.b_mu, jnp.log(1 + jnp.exp(self.b_rho)))
        log_post = jnp.sum(w_log_post) + jnp.sum(b_log_post)
        if train:
            return jnp.dot(x, w.T) + b, log_prior, log_post
        return jnp.dot(x, w.T) + b


def _pair(kernel_size):
    if isinstance(kernel_size, tuple):
        return kernel_size
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)


class BayesConv2d(nn.Module):
    """
    Conv Layer of BayesianNN.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (int | tuple[int, int]): size of the conv kernel
        stride (int | tuple[int, int], optional): stride of the convolution
        padding (int | tuple[int, int] | str, optional): padding added to all four sides of the input
        dilation (int | tuple[int, int], optional): spacing between kernel elements
        groups (int, optional): number of blocked connections from input channels to output channels
        bias (bool, optional): if True, adds a learnable bias to the output
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int], str] = 0
    dilation: Union[int, Tuple[int, int]] = 1
    groups: int = 1
    bias: bool = False
    prior_mean: float = 0.0
    prior_var: float = 1.0

    def setup(self):
        self.w_mu = self.param('w_mu', nn.initializers.zeros,
                               (self.out_channels, self.in_channels, *_pair(self.kernel_size)))
        self.w_rho = self.param('w_rho', nn.initializers.zeros,
                                (self.out_channels, self.in_channels, *_pair(self.kernel_size)))
        if self.bias:
            self.b_mu = self.param('b_mu', nn.initializers.zeros, (self.out_channels))
            self.b_rho = self.param('b_rho', nn.initializers.zeros, (self.out_channels))

    def __call__(self, x, train=False):
        # sample weights

        w_epsilon = np.random.normal(0, 1, self.w_mu.shape)
        w = self.w_mu + jnp.log(1 + jnp.exp(self.w_rho)) * w_epsilon

        if self.bias:
            b_epsilon = np.random.normal(0, 1, self.b_mu.shape)
            b = self.b_mu + jnp.log(1 + jnp.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias

        x = jax.lax.conv_general_dilated(x, w, _pair(self.stride), ((_pair(self.padding)[0], _pair(self.padding)[1]),
                                                                    (_pair(self.padding)[0], _pair(self.padding)[1])),
                                         rhs_dilation=_pair(self.dilation), feature_group_count=self.groups)

        if self.bias:
            b = b.reshape(-1, 1, 1)
            x += b

        if not train:
            return x

        w_log_prior = stats.norm.logpdf(w, self.prior_mean, jnp.sqrt(self.prior_var))
        log_prior = jnp.sum(w_log_prior)
        if self.bias:
            b_log_prior = stats.norm.logpdf(b, self.prior_mean, jnp.sqrt(self.prior_var))
            log_prior += jnp.sum(b_log_prior)

        w_log_post = stats.norm.logpdf(w, self.w_mu, jnp.log(1 + jnp.exp(self.w_rho)))
        log_post = jnp.sum(w_log_post)
        if self.bias:
            b_log_post = stats.norm.logpdf(b, self.b_mu, jnp.log(1 + jnp.exp(self.b_rho)))
            log_post += jnp.sum(b_log_post)

        return x, log_prior, log_post


class BayesTransConv2d(nn.Module):
    """
    Transposed Conv Layer of BayesianNN.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (int | tuple[int, int]): size of the conv kernel
        stride (int | tuple[int, int], optional): stride of the convolution
        padding (int | tuple[int, int], optional): padding added to all four sides of the input
        output_padding (int | tuple[int, int], optional): padding added to all four sides of the output
        dilation (int | tuple[int, int], optional): spacing between kernel elements
        groups (int, optional): number of blocked connections from input channels to output channels
        bias (bool, optional): if True, adds a learnable bias to the output.
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int]] = 0
    output_padding: Union[int, Tuple[int, int]] = 0
    dilation: Union[int, Tuple[int, int]] = 1
    groups: int = 1
    bias: bool = False
    prior_mean: float = 0.0
    prior_var: float = 1.0

    def setup(self):
        self.w_mu = self.param('w_mu', nn.initializers.zeros,
                               (self.in_channels, self.out_channels, *_pair(self.kernel_size)))
        self.w_rho = self.param('w_rho', nn.initializers.zeros,
                                (self.in_channels, self.out_channels, *_pair(self.kernel_size)))
        if self.bias:
            self.b_mu = self.param('b_mu', nn.initializers.zeros, (self.out_channels))
            self.b_rho = self.param('b_rho', nn.initializers.zeros, (self.out_channels))

    def __call__(self, x, train=False):
        # sample weights

        w_epsilon = np.random.normal(0, 1, self.w_mu.shape)
        w = self.w_mu + jnp.log(1 + jnp.exp(self.w_rho)) * w_epsilon

        if self.bias:
            b_epsilon = np.random.normal(0, 1, self.b_mu.shape)
            b = self.b_mu + jnp.log(1 + jnp.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias

        _kernel_size = _pair(self.kernel_size)
        _dilation = _pair(self.dilation)
        _stride = _pair(self.stride)
        _padding = _pair(self.padding)
        _outpadding = _pair(self.output_padding)

        padding_before = lambda i : _kernel_size[i] - 1 + (_kernel_size[i] - 1) * (_dilation[i] - 1) - _padding[i]
        padding_after = lambda i : padding_before(i) + _outpadding[i]

        x = jax.lax.conv_transpose(x,
                                   w,
                                   padding=((padding_before(0), padding_after(0)),
                                            (padding_before(1), padding_after(1))),
                                   strides=(_stride[0], _stride[1]),
                                   rhs_dilation=(_dilation[0], _dilation[1]),
                                   dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                                   transpose_kernel=True)
        if self.bias:
            b = b.reshape(-1, 1, 1)
            x += b


        if not train:
            return x

        w_log_prior = stats.norm.logpdf(w, self.prior_mean, jnp.sqrt(self.prior_var))
        log_prior = jnp.sum(w_log_prior)
        if self.bias:
            b_log_prior = stats.norm.logpdf(b, self.prior_mean, jnp.sqrt(self.prior_var))
            log_prior += jnp.sum(b_log_prior)

        w_log_post = stats.norm.logpdf(w, self.w_mu, jnp.log(1 + jnp.exp(self.w_rho)))
        log_post = jnp.sum(w_log_post)
        if self.bias:
            b_log_post = stats.norm.logpdf(b, self.b_mu, jnp.log(1 + jnp.exp(self.b_rho)))
            log_post += jnp.sum(b_log_post)

        return x, log_prior, log_post