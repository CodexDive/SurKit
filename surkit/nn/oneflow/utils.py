#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import math
from numbers import Number, Real
from typing import Any, Dict

import oneflow as flow


def is_tensor_like(inp):
    return type(inp) is flow.Tensor

def broadcast_all(*values):
    if not all(is_tensor_like(v) or isinstance(v, Number)
               for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number, '
                         'torch.Tensor or objects implementing __torch_function__.')
    if not all(is_tensor_like(v) for v in values):
        options: Dict[str, Any] = dict(dtype=flow.get_default_dtype())
        for value in values:
            if isinstance(value, flow.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [v if is_tensor_like(v) else flow.tensor(v, **options)
                      for v in values]
        return flow.broadcast_tensors(*new_values)
    return flow.broadcast_tensors(*values)

class _Real:
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """
    is_discrete = False  # Default to continuous.
    event_dim = 0  # Default to univariate.
    def check(self, value):
        return value == value  # False for NANs.

class Normal(flow.distributions.Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Normal(flow.tensor([0.0]), flow.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    support = _Real()
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = flow.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=flow.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
        """
        shape = self._extended_shape(sample_shape)
        with flow.no_grad():
            return flow.normal(self.loc, self.scale, shape)

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at value.
        """
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))