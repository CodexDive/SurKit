# #! /usr/bin/python
# # -*- coding: utf-8 -*-

import os
import sys

backend_name = None
supported_backends = [
    "torch",
    "oneflow",
    "jax",
]


# Set backend based on ES_BACKEND.
if 'SRK_BACKEND' in os.environ:
    backend = os.environ['SRK_BACKEND']
    if backend:
        backend_name = backend

# import backend functions

if not backend_name:
    print("Backend not specified, search for available backends.")
    for backend in supported_backends:
        try:
            __import__(backend)
            backend_name = backend
            break
        except ImportError:
            continue
    if not backend_name:
        raise ValueError("No supported backends found. Please install one of %s" % (', '.join(supported_backends)))


if backend_name in ['torch', 'pytorch']:
    backend_name = 'pytorch'
    from .pytorch_bkd import *
    BACKEND_VERSION = torch.__version__
    sys.stdout.write('Using PyTorch backend.\n')

elif backend_name == 'oneflow':
    from .oneflow_bkd import *
    import oneflow as flow
    BACKEND_VERSION = flow.__version__
    sys.stdout.write('Using OneFlow backend.\n')

elif backend_name == 'jax':
    from .jax_bkd import *
    BACKEND_VERSION = jax.__version__
    sys.stdout.write('Using JAX backend.\n')

else:
    raise NotImplementedError("Backend %s is not supported" % backend_name)
