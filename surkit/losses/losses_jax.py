#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import jax.numpy as jnp
from jax.scipy import stats


def get(loss, reduction='mean'):
    """
    Return a loss function based on the given string, or return the given callable loss function

    Args:
        loss (str | function): name of a loss function or a callable jax loss function
        reduction (str): reduction applied to the output

    Returns:
        function: a callable loss function
    """
    loss_dict = {
        "mean absolute error": L1Loss(reduction=reduction),
        "mae": L1Loss(reduction=reduction),
        "mean squared error": MSELoss(reduction=reduction),
        "mse": MSELoss(reduction=reduction),
        "gaussiannll": NLLLoss(reduction=reduction),
    }

    if isinstance(loss, str):
        return loss_dict[loss.lower()]

    if callable(loss):
        return loss

def MSELoss(reduction="mean"):
    if reduction == "mean":
        return lambda predictions, target: ((predictions - target) ** 2).mean()
    elif reduction == "sum":
        return lambda predictions, target: ((predictions - target) ** 2).sum()
    return lambda predictions, target: ((predictions - target) ** 2)

def L1Loss(reduction="mean"):
    if reduction == "mean":
        return lambda predictions, target: (predictions - target).mean()
    elif reduction == "sum":
        return lambda predictions, target: (predictions - target).sum()
    return lambda predictions, target: predictions - target

def NLLLoss(reduction="mean"):
    if reduction == "mean":
        return lambda y, mean, var : (jnp.log(var) + (y - mean) ** 2 / (2 * var)).mean()
    elif reduction == "sum":
        return lambda y, mean, var : (jnp.log(var) + (y - mean) ** 2 / (2 * var)).sum()
    return lambda y, mean, var : jnp.log(var) + (y - mean) ** 2 / (2 * var)

def elbo(x, y, samples, net, params):
    """
    Calculate negative evidence lower bound for bayes nn variational inference

    Args:
        x (tensor): input
        y (tensor): target
        samples (int): number of nn samples
        net (module): bayes nn
        params (): parameter of nn
    Returns:
        tensor: loss, negative elbo
    """
    outs = []
    log_priors = []
    log_posts = []
    log_likes = []


    for i in range(samples):
        try:
            out, prior, post = net.apply(params, x, True, mutable=['batch_stats'])
        except:
            out, prior, post = net.apply(params, x, True)
        outs.append(out)
        log_priors.append(prior)  # get log prior
        log_posts.append(post)  # get log variational posterior
        log_likes.append(stats.norm.logpdf(y, out, net.noise_tol).sum()) # calculate the log likelihood
    # calculate monte carlo estimate of prior posterior and likelihood
    log_prior = jnp.array(log_priors).mean()
    log_post = jnp.array(log_posts).mean()
    log_like = jnp.array(log_likes).mean()
    # calculate the negative elbo (which is our loss function)

    loss = log_post - log_prior - log_like
    return loss