#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from .. import backend as bkd

from ..utils.parse import eval_with_vars


def parse_pde(equation, inputs: dict, constants: dict, out_dic: dict, dic: dict):
    """
    Evaluate the left and right parts of the equation evaluated using the provided dictionaries and governing equation string.

    Args:
        equation (str): differential equation
        inputs (dict[str, Tensor]): dict with input features' names and their values
        constants (dict[str, int | float]): dict with constants' names and their values
        out_dic (dict[str, Tensor]): dict with output features' names and their values
        dic (dict[str, Tensor]): dict with gradients' names and their values

    Returns:
        tuple[Tensor, Tensor]: the values of the left and right parts
    """
    ll = eval_with_vars(equation[0], inputs, constants, out_dic, dic)
    rr = eval_with_vars(equation[1], inputs, constants, out_dic, dic)

    if not isinstance(rr, bkd.Tensor):
        rr = bkd.zeros_like(ll) + rr
        if not isinstance(rr, bkd.Tensor):
            rr = bkd.np_to_tensor(rr)

    return ll, rr
