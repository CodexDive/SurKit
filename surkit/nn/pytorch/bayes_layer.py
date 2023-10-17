#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.modules.utils import _pair


class BayesLinear(nn.Module):
    """
    Linear Layer of BayesianNN.

    Args:
        in_d (int): input length
        out_d (int): output length
        prior_mean (float, optional): prior distribution mean
        prior_var (float, optional): prior distribution variance
    """
    def __init__(self, in_d, out_d, prior_mean=0., prior_var=1.):
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.log_post = None
        self.b_post = None
        self.w_post = None
        self.log_prior = None
        self.in_d = in_d
        self.out_d = out_d

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(out_d, in_d))
        self.w_rho = nn.Parameter(torch.zeros(out_d, in_d))

        # initialize mu and rho parameters for the layer's bias

        self.b_mu = nn.Parameter(torch.zeros(out_d))
        self.b_rho = nn.Parameter(torch.zeros(out_d))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all the weights and biases
        self.prior = torch.distributions.Normal(prior_mean, np.sqrt(prior_var))

    def forward(self, x):
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(self.w_mu.device)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(self.b_mu.device)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with
        # respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(x, self.w, self.b)


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
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, prior_mean=0., prior_var=1.):
        # initialize layers
        super().__init__()
        # set input and output dimensions

        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.groups = groups

        self.log_prior = None
        self.log_post = None
        self.w_post = None
        if self.bias:
            self.b_post = None

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(out_channels, in_channels, *_pair(kernel_size)))
        self.w_rho = nn.Parameter(torch.zeros(out_channels, in_channels, *_pair(kernel_size)))
        #
        # initialize mu and rho parameters for the layer's bias
        if self.bias:
            self.b_mu = nn.Parameter(torch.zeros(out_channels))
            self.b_rho = nn.Parameter(torch.zeros(out_channels))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        #
        # initialize prior distribution for all the weights and biases
        self.prior = torch.distributions.Normal(prior_mean, np.sqrt(prior_var))

    def forward(self, x):
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(self.w_mu.device)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        if self.bias:
            b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(self.b_mu.device)
            self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        self.log_prior = torch.sum(w_log_prior)
        if self.bias:
            b_log_prior = self.prior.log_prob(self.b)
            self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with
        # respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum()
        if self.bias:
            self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
            self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.conv2d(x, weight=self.w, bias=self.b, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=False, prior_mean=0., prior_var=1.):
        # initialize layers
        super().__init__()
        # set input and output dimensions

        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups

        self.log_prior = None
        self.log_post = None
        self.w_post = None
        if self.bias:
            self.b_post = None

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(in_channels, out_channels, *_pair(kernel_size)))
        self.w_rho = nn.Parameter(torch.zeros(in_channels, out_channels, *_pair(kernel_size)))
        #
        # initialize mu and rho parameters for the layer's bias
        if self.bias:
            self.b_mu = nn.Parameter(torch.zeros(out_channels))
            self.b_rho = nn.Parameter(torch.zeros(out_channels))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        #
        # initialize prior distribution for all the weights and biases
        self.prior = torch.distributions.Normal(prior_mean, prior_var)

    def forward(self, x):
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(self.w_mu.device)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        if self.bias:
            b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(self.b_mu.device)
            self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        self.log_prior = torch.sum(w_log_prior)
        if self.bias:
            b_log_prior = self.prior.log_prob(self.b)
            self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with
        # respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum()
        if self.bias:
            self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
            self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.conv_transpose2d(x, weight=self.w, bias=self.b, stride=self.stride, padding=self.padding,
                                  output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)
