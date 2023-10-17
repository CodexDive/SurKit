#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow

def transform(trans, x):
    """
    The affine transform of FNN.

    Args:
        trans (str): transform type
        x (tensor): input

    Returns:
        tensor: transformed input

    """

    if trans == "sin":
        return flow.sin(x)
    if trans == "cos":
        return flow.cos(x)
    if trans == "pow2":
        return flow.pow(x, 2)
    if trans == "pow3":
        return flow.pow(x, 3)
    if trans == "exp":
        return flow.exp(x)
