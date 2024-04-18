#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get', 'elbo']


def get(loss, reduction='mean', smooth=1.):
    """
    Return a loss function based on the given string, or return the given callable loss function

    Args:
        loss (str | function): name of a loss function or a callable pytorch loss function
        reduction (str): reduction applied to the output

    Returns:
        function: a callable loss function
    """
    loss_dict = {
        "mean absolute error": nn.L1Loss(reduction=reduction),
        "mae": nn.L1Loss(reduction=reduction),
        "mean squared error": nn.MSELoss(reduction=reduction),
        "mse": nn.MSELoss(reduction=reduction),
        "gaussiannll": nn.GaussianNLLLoss(reduction=reduction),
        "dice": DiceLoss(smooth=smooth),
        "bce": nn.BCELoss(reduction=reduction),
        "binary cross entropy": nn.BCELoss(reduction=reduction),
        "dice and bce": DC_and_BCE(smooth=smooth),
    }

    if isinstance(loss, str):
        return loss_dict[loss.lower()]

    if callable(loss):
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
        return 1 - (2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth)

class DC_and_BCE(nn.Module):
    def __init__(self, smooth: float = 1.) -> None:
        super(DC_and_BCE, self).__init__()
        self.smooth = smooth
        

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = input.view(-1)
        targets = target.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE * bce_weight + dice_loss * (1 - bce_weight)

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
    log_priors = torch.zeros(samples)
    log_posts = torch.zeros(samples)
    log_likes = torch.zeros(samples)

    for i in range(samples):
        out[i] = net(x).reshape(-1)  # make predictions
        log_priors[i] = net.log_prior()  # get log prior
        log_posts[i] = net.log_post()  # get log variational posterior
        log_likes[i] = torch.distributions.Normal(out[i], net.noise_tol).log_prob(
            y.reshape(-1)).sum()  # calculate the log likelihood
    # calculate monte carlo estimate of prior posterior and likelihood
    log_prior = log_priors.mean()
    log_post = log_posts.mean()
    log_like = log_likes.mean()
    # calculate the negative elbo (which is our loss function)

    loss = log_post - log_prior - log_like
    return loss
