#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import jax.numpy as jnp

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
        return jnp.sin(x)
    if trans == "cos":
        return jnp.cos(x)
    if trans == "pow2":
        return jnp.power(x, 2)
    if trans == "pow3":
        return jnp.power(x, 3)
    if trans == "exp":
        return jnp.exp(x)
