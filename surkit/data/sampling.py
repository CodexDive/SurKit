#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np


def get(sampler):
    """
    Return a sampler based on the given string

    Args:
        sampler (str): "random" or "uniform"

    Returns:
        function: a pre-defined sampler function

    """

    sampler_dict = {
        "random": random_sampler,
        "uniform": uniform_sampler,
    }

    return sampler_dict[sampler]


def random_sampler(lower=0, upper=1, n=1):
    """
    A random sampler

    Args:
        lower (float | ndarray | Iterable | int | None): lower bound
        upper (float | ndarray | Iterable | int | None): upper bound
        n (int | Iterable | tuple[int] | None]): sample size

    Returns:
        numpy.ndarray: a set of sampling data
    """
    # return np.random.random(n) * (upper - lower) + lower
    return np.random.uniform(low=lower, high=upper, size=n)


def uniform_sampler(lower=0, upper=1, n=1):
    """
    A uniform sampler

    Args:
        lower (float | ndarray | Iterable | int | None): lower bound
        upper (float | ndarray | Iterable | int | None): upper bound
        n (int | Iterable | tuple[int] | None]): sample size

    Returns:
        numpy.ndarray: a set of sampling data
    """
    return np.expand_dims(np.linspace(lower, upper, n, endpoint=True), 1)
    # return np.linspace(lower, upper, n, endpoint=True)

def weight_2d(x, y, xrange, yrange, dt):
    """
    Assign weight according to 2-dimensional coordinates

    Args:
        x (float): the value of the first dimension
        y (float): the value of the second dimension
        xrange (list[float, float] | tuple[float, float]): the boundary of the first dimension
        yrange (list[float, float] | tuple[float, float]): the boundary of the second dimension
        dt (float): empirical parameters controlling the extent of the weighted area

    Returns:
        float: weight
    """
    x_std = (x - xrange[0]) / (xrange[1] - xrange[0])
    y_std = (y - yrange[0]) / (yrange[1] - yrange[0])

    dlx = min(x_std, 1 - x_std)
    dly = min(y_std, 1 - y_std)
    dl = min(dlx, dly)
    if dl == 0:
        if dl > 0.5 - dt:
            return dl / (0.5 - dt)
        else:
            return 1
    else:
        if dlx == 0:
            return 1 - 4 * (0.5 - dly) ** 2
        elif dly == 0:
            return 1 - 4 * (0.5 - dlx) ** 2

def weight_1d(x, xrange):
    """
    Assign weight according to 1-dimensional coordinates

    Args:
        x (float): value
        xrange (list[float, float] | tuple[float, float]): boundary

    Returns:
        float: weight
    """
    x_std = (x - xrange[0]) / (xrange[1] - xrange[0])
    return 1 - 4 * (0.5 - x_std) ** 2