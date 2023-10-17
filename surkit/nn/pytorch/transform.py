#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch

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
        return torch.sin(x)
    if trans == "cos":
        return torch.cos(x)
    if trans == "pow2":
        return torch.pow(x, 2)
    if trans == "pow3":
        return torch.pow(x, 3)
    if trans == "exp":
        return torch.exp(x)
