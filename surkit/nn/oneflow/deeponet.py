#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
import oneflow.nn as nn

from .fnn import FNN

class DeepONet(nn.Module):

    def __init__(
            self,
            layer_size_branch,
            layer_size_trunk,
            activation,
            in_d_branch,
            in_d_trunk,
            mid_d,
            initializer="He uniform",
    ):
        super(DeepONet, self).__init__()

        self.branch_net = FNN(layer_size_branch, activation, in_d_branch, mid_d, initializer)
        self.trunk_net = FNN(layer_size_trunk, activation, in_d_trunk, mid_d, initializer)
        self.b = flow.tensor(flow.zeros(1), requires_grad=True)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk = self.trunk_net(x_trunk)
        y = flow.einsum("bi,ni->bn", y_branch, y_trunk)
        y += self.b
        return y