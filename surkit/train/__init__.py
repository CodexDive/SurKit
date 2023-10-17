#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os

from ..backend import backend_name

if os.environ.get("READTHEDOCS") == "True":
    from . import train_pytorch
    from . import train_oneflow
    from . import train_jax


if backend_name == 'pytorch':
    from .train_pytorch import *
elif backend_name == 'oneflow':
    from .train_oneflow import *
elif backend_name == 'jax':
    from .train_jax import *