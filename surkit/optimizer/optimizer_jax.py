#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import optax



def get(optimizer):
    """
    Return an optimizer based on the given string, or return the callable optimizer function

    Args:
        optimizer (str or callable): Optimizer name or optax optimizer.

    Returns:
        callable: Optimizer class.
    """

    optimizer_dict = {
        "adam": optax.adam,
        # "l-bfgs-b": optax.LBFGS,
        # "lbfgs": optax.LBFGS,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "adagrad": optax.adagrad,
    }

    if isinstance(optimizer, str):
        return optimizer_dict[optimizer.lower()]

    if callable(optimizer):
        return optimizer

    raise ValueError("optimizer must be either a string or a callable")
