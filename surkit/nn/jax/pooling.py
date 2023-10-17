#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from typing import Union, Tuple

import flax.linen as nn
import jax.numpy as jnp


def _pair(i):
    if isinstance(i, tuple):
        return i
    if isinstance(i, int):
        return i, i

class AvgPool2d(nn.Module):
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = None
    padding: Union[int, Tuple[int, int]] = 0

    def setup(self):
        self.kernel = _pair(self.kernel_size)
        if self.stride:
            self.strides = _pair(self.stride)
        else:
            self.strides = _pair(self.kernel_size)
        self.pad = ((_pair(self.padding)[0], _pair(self.padding)[0]), (_pair(self.padding)[1], _pair(self.padding)[1]))
        
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        return jnp.transpose(nn.avg_pool(x, window_shape=self.kernel, strides=self.strides, padding=self.pad), (0, 3, 1, 2))


class MaxPool2d(nn.Module):
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = None
    padding: Union[int, Tuple[int, int]] = 0

    def setup(self):
        self.kernel = _pair(self.kernel_size)
        if self.stride:
            self.strides = _pair(self.stride)
        else:
            self.strides = _pair(self.kernel_size)
        self.pad = ((_pair(self.padding)[0], _pair(self.padding)[0]), (_pair(self.padding)[1], _pair(self.padding)[1]))

    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        return jnp.transpose(nn.max_pool(x, window_shape=self.kernel, strides=self.strides, padding=self.pad), (0, 3, 1, 2))

class LPPool2d(nn.Module):
    power : float
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = None
    padding = None

    def f(self, a, b):
        return (a ** self.power + b ** self.power) ** (1 / self.power)

    def setup(self):
        self.kernel = _pair(self.kernel_size)
        if self.stride:
            self.strides = _pair(self.stride)
        else:
            self.strides = _pair(self.kernel_size)


    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.pool(x, 0., reduce_fn=self.f, window_shape=self.kernel, strides=self.strides, padding=((0, 0), (0, 0)))
        return jnp.transpose(x, (0, 3, 1, 2))

def get(pool):
    """
    Return a pooling layer based on the given string, or return the given pooling layer

    Args:
        pool (str | nn.Module): name of a pooling layer or a jax pooling layer

    Returns:
        nn.Module: a pooling layer

    """

    pool_dict = {
        'average': AvgPool2d,
        'max': MaxPool2d,
        'lp': LPPool2d,
    }

    if isinstance(pool, str):
        return pool_dict[pool.lower()]

    if callable(pool):
        return pool
