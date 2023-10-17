#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import flax.linen as nn
import jax.image


class Dropout2D(nn.Module):
    p: float

    def setup(self):
        self.main = nn.Dropout(self.p, broadcast_dims=[2, 3])

    def __call__(self, x, train=False):
        x = self.main(x, deterministic=not train)
        return x


class Dropout(nn.Module):
    p: float

    def setup(self):
        self.main = nn.Dropout(self.p)

    def __call__(self, x, train=False):
        x = self.main(x, deterministic=not train)
        return x


class Upsample(nn.Module):
    scale_factor : float
    mode : str = 'bilinear'
    align_corners : bool = True

    def __call__(self, x):
        outshape = (x.shape[0], x.shape[1], x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor)
        return jax.image.resize(image=x, shape=outshape, method=self.mode)

class Flatten(nn.Module):

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)