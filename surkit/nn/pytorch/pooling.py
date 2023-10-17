#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch.nn as nn


def get(pool):
    """
    Return a pooling layer based on the given string, or return the given pooling layer

    Args:
        pool (str | nn.Module): name of a pooling layer or a pytorch pooling layer

    Returns:
        nn.Module: a pooling layer

    """

    pool_dict = {
        'average': nn.AvgPool2d,
        'max': nn.MaxPool2d,
        'lp': nn.LPPool2d,
    }

    if isinstance(pool, str):
        return pool_dict[pool.lower()]

    if callable(pool):
        return pool
