#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow.nn as nn

__all__ = ['get']


def get(initializer):
    """
    Return an initializer based on the given string, or return the given callable initializer

    Args:
        initializer (str | callable): name of an initializer or a callable oneflow initializer

    Returns:
        callable: a callable activation function

    """
    initializer_dict = {
        "glorot normal": nn.init.xavier_normal_,
        "glorot uniform": nn.init.xavier_uniform_,
        "xavier normal": nn.init.xavier_normal_,
        "xavier uniform": nn.init.xavier_uniform_,
        "he normal": nn.init.kaiming_normal_,
        "he uniform": nn.init.kaiming_uniform_,
        "kaiming normal": nn.init.kaiming_normal_,
        "kaiming uniform": nn.init.kaiming_uniform_,
        "zeros": nn.init.zeros_,
    }

    if isinstance(initializer, str):
        return initializer_dict[initializer.lower()]

    if callable(initializer):
        return initializer
