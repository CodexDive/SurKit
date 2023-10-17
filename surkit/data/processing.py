#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from .. import backend as bkd


def np_to_tensor(np):
    """

    Args:
        np (list[numpy.ndarray] | numpy.ndarray): data that needs to be converted to tensor

    Returns:
        list[tensor] | tensor :  tensor format data
    """
    if isinstance(np, list):
        return [bkd.np_to_tensor(n, requires_grad=True) for n in np]
    return bkd.np_to_tensor(np, requires_grad=True)