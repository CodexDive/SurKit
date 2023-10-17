#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from .. import backend as bkd

# 改；没考虑公式中有常量的情况
def parse_icbc2d(condition: str, inputs: dict, constants: dict, outs: dict, icbc_intermediate_dict:dict):
    """
    Currently only physics informed dense_ed use this function. Because the boundary of the field in image-form is
    simply the outermost pixels, only compare the output images' the outermost pixels and the target value.

    Args:
        condition (str): condition formula defines the outputs given certain inputs, i.e. "$u@left$ = 0" means the leftmost part of this
         two-dimensional field (possibly in the case of x = 0), u equals 0
        inputs (dict[str, bkd.Tensor]): dictionary of input data, i.e. {'A': TensorA, 'B': TensorB}
        constants (dict[str, float]): dictionary of constants that will be used in the calculation process, i.e. {'pi': 3.14}
        outs (dict[str, bkd.Tensor]): dictionary of the model's output, i.e. {'U': TensorU}
        icbc_intermediate_dict (dict[str, bkd.Tensor]): dictionary of values to be derived from the boundary conditions

    Returns:
        list[tuple[bkd.Tensor]]: return a list of prediction and ground truth pairs under initial & boundary conditions
    """

    def get_value(value: str):
        name = value[value.find("'") + 1 : value.find("@")]
        candidate = outs[name]
        if 'left' in value:
            return candidate[:, :, :, 0]
        elif 'right' in value:
            return candidate[:, :, :, -1]
        elif 'top' in value:
            return candidate[:, :, 0, :]
        elif 'down' in value:
            return candidate[:, :, -1, :]

    lhs, rhs = condition.replace(' ', '').split('=')

    if '@' in lhs:
        left = get_value(lhs)
        right = bkd.zeros_like(left) + float(rhs)
    else:
        right = get_value(lhs)
        left = bkd.zeros_like(right) + float(lhs)
    return left, right
