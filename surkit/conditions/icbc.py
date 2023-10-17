#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from .. import backend as bkd
from ..pde.pde import parse_pde
from ..utils.parse import get_grad, eval_with_vars

def parse_icbc(condition: str, inputs: dict, constants: dict, outs: list, icbc_intermediate_dict: dict, net):
    """
    Calculate initial condition & boundary condition losses of pde

    Args:
        condition (str): condition formula defines the outputs given certain inputs, i.e. "$u$ = - sin($pi$ * $x$) @ $t$ = 0" means when t = 0, u = -sin(pi*x)
        inputs (dict[str, tensor]): dictionary of input data, i.e. {'A': TensorA, 'B': TensorB}
        constants (dict[str, float]): dictionary of constants that will be used in the calculation process, i.e. {'pi': 3.14}
        outs (dict[str, tensor]): dictionary of the model's output, i.e. {'U': TensorU}
        icbc_intermediate_dict (dict[str, tensor]): dictionary of values to be derived from the boundary conditions
        net (module): the surrogate model

    Returns:
        list[tuple[tensor]]: return a list of prediction and ground truth pairs under initial & boundary conditions

    """
    conds, whens = condition
    in_dict = inputs.copy()
    for when in whens:
        wl, wr = when
        in_dict[wl[8:-2]] = bkd.zeros_like(in_dict[wl[8:-2]]) + eval_with_vars(wr, inputs, constants, {}, {})

    in_ = bkd.cat(list(in_dict.values()), dim=1)
    out_ = bkd.forward(net, in_)

    out_dic = {}
    for index, out in enumerate(outs):
        out_dic[out] = bkd.unsqueeze(out_[:, index], dim=1)

    for key in icbc_intermediate_dict.keys():
        icbc_intermediate_dict[key] = get_grad(key, in_dict, icbc_intermediate_dict, out_dic)

    value_target_pairs = []
    for cond in conds:
        value_target_pairs.append(
            parse_pde(cond, in_dict, constants, out_dic, icbc_intermediate_dict)
        )
    return value_target_pairs

