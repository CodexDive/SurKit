#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os

from ..backend import backend_name

if os.environ.get("READTHEDOCS") == "True":
    from . import pytorch
    from . import oneflow
    from . import jax


if backend_name == 'pytorch':
    from .pytorch import *
elif backend_name == 'oneflow':
    from .oneflow import *
elif backend_name == 'jax':
    from .jax import *
#
# def _load_backend(mod_name):
#     mod = importlib.import_module(".%s" % mod_name, __name__)
#     thismod = sys.modules[__name__]
#     for api, obj in mod.__dict__.items():
#         setattr(thismod, api, obj)
#
# _load_backend(backend_name)