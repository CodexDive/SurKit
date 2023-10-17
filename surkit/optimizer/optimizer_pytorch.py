#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch.optim



def get(optimizer):
    """
    Return an optimizer based on the given string, or return the callable optimizer function

    Args:
        optimizer (str or callable): Optimizer name or callable pytorch optimizer.

    Returns:
        callable: Optimizer class.
    """

    optimizer_dict = {
        "adam": torch.optim.Adam,
        "l-bfgs-b": torch.optim.LBFGS,
        "lbfgs": torch.optim.LBFGS,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }

    if isinstance(optimizer, str):
        return optimizer_dict[optimizer.lower()]

    if callable(optimizer):
        return optimizer

    raise ValueError("optimizer must be either a string or a callable")