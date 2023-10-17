#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import jax

__all__ = ['get']


def get(initializer):
    """
    Return an initializer based on the given string, or return the given callable initializer

    Args:
        initializer (str | callable): name of an initializer or a callable jax initializer

    Returns:
        callable: a callable initializer function

    """
    initializer_dict = {
        "glorot normal": jax.nn.initializers.glorot_normal(),
        "glorot uniform": jax.nn.initializers.glorot_uniform(),
        "xavier normal": jax.nn.initializers.glorot_normal(),
        "xavier uniform": jax.nn.initializers.glorot_uniform(),
        "he normal": jax.nn.initializers.kaiming_normal(),
        "he uniform": jax.nn.initializers.kaiming_uniform(),
        "kaiming normal": jax.nn.initializers.kaiming_normal(),
        "kaiming uniform": jax.nn.initializers.kaiming_uniform(),
        "zeros": jax.nn.initializers.zeros,
    }

    if isinstance(initializer, str):
        return initializer_dict[initializer.lower()]

    if callable(initializer):
        return initializer
