#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
import oneflow.nn as nn

__all__ = ['get', 'NLLloss']

from surkit.nn.oneflow.utils import Normal


def get(loss, reduction='mean'):
    """
    Return a loss function based on the given string, or return the given callable loss function

    Args:
        loss (str | function): name of a loss function or a callable oneflow loss function
        reduction (str): reduction applied to the output

    Returns:
        function: a callable loss function
    """
    loss_dict = {
        "mean absolute error": nn.L1Loss(reduction=reduction),
        "mae": nn.L1Loss(reduction=reduction),
        "mean squared error": nn.MSELoss(reduction=reduction),
        "mse": nn.MSELoss(reduction=reduction),
        "gaussiannll": GaussianNLLLoss(reduction=reduction),
    }

    if isinstance(loss, str):
        return loss_dict[loss.lower()]

    if callable(loss):
        return loss

def elbo(x, y, samples, net):
    """
    Calculate negative evidence lower bound for bayes nn variational inference

    Args:
        x (tensor): input
        y (tensor): target
        samples (int): number of nn samples
        net (module): bayes nn

    Returns:
        tensor: loss, negative elbo
    """

    out = [None for _ in range(samples)]
    log_priors = flow.zeros(samples)
    log_posts = flow.zeros(samples)
    log_likes = flow.zeros(samples)

    for i in range(samples):
        out[i] = net(x).reshape(-1)  # make predictions
        log_priors[i] = net.log_prior()  # get log prior
        log_posts[i] = net.log_post()  # get log variational posterior
        log_likes[i] = Normal(out[i], net.noise_tol).log_prob(
            y.reshape(-1)).sum()  # calculate the log likelihood
    # calculate monte carlo estimate of prior posterior and likelihood
    log_prior = log_priors.mean()
    log_post = log_posts.mean()
    log_like = log_likes.mean()
    # calculate the negative elbo (which is our loss function)

    loss = log_post - log_prior - log_like
    return loss

class _Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        assert reduction in ["none", "mean", "sum"]
        self.reduction = reduction

class GaussianNLLLoss(_Loss):

    def __init__(self, reduction: str = "mean") -> None:
        super(GaussianNLLLoss, self).__init__(reduction)

    def forward(self, target, mean, var):
        return NLLloss(target, mean, var, reduction="mean")


def NLLloss(y, mean, var, reduction="mean"):
    if reduction == "mean":
        return (flow.log(var) + (y - mean) ** 2 / (2 * var)).mean()
    if reduction == "sum":
        return (flow.log(var) + (y - mean) ** 2 / (2 * var)).sum()
    if reduction == "none":
        return flow.log(var) + (y - mean) ** 2 / (2 * var)