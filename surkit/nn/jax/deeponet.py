#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from typing import Sequence

import jax.numpy
from flax import linen as nn
from flax.linen import compact

from .fnn import FNN

class DeepONet(nn.Module):
    layer_size_branch: Sequence[int]
    layer_size_trunk: Sequence[int]
    activation: str
    in_d_branch: int
    in_d_trunk: int
    mid_d: int = 128
    initializer: str = "He uniform"
    def setup(self):
        self.branch_net = FNN(self.layer_size_branch, self.activation, self.in_d_branch, self.mid_d, self.initializer)
        self.trunk_net = FNN(self.layer_size_trunk, self.activation, self.in_d_trunk, self.mid_d, self.initializer)
        self.b = self.param('excitation_block', jax.numpy.zeros, 1)

    def __call__(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk = self.trunk_net(x_trunk)
        y = jax.numpy.einsum("bi,ni->bn", y_branch, y_trunk)
        y += self.b
        return y