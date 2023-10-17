#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os

from ..backend import backend_name

if os.environ.get("READTHEDOCS") == "True":
    from . import losses_pytorch
    from . import losses_oneflow
    from . import losses_jax


if backend_name == 'pytorch':
    from .losses_pytorch import *
elif backend_name == 'oneflow':
    from .losses_oneflow import *
elif backend_name == 'jax':
    from .losses_jax import *
