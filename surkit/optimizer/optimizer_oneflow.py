#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow.optim


def get(optimizer):
    """
    Return an optimizer based on the given string, or return the callable optimizer function

    Args:
        optimizer (str or callable): Optimizer name or callable oneflow optimizer.

    Returns:
        callable: Optimizer class.
    """

    optimizer_dict = {
        "adam": oneflow.optim.Adam,
        # oneflow文档中有，但未实装
        # "l-bfgs-b": oneflow.optim.LBFGS,
        # "lbfgs": oneflow.optim.LBFGS,
        "sgd": oneflow.optim.SGD,
        "rmsprop": oneflow.optim.RMSprop,
        "adagrad": oneflow.optim.Adagrad,
    }

    if isinstance(optimizer, str):
        return optimizer_dict[optimizer.lower()]

    if callable(optimizer):
        return optimizer

    raise ValueError("optimizer must be either a string or a callable")