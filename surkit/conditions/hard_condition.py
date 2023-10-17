#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from ..utils.parse import eval_with_vars


def convert(solutions: list, inputs: dict, constants: dict, outs: dict):
    """
    Perform solutions that just satisfies IC/BC which given by expert knowledge.

    Args:
        solutions (list[str]): list of solutions given by expert knowledge
        inputs (dict[str, tensor]): dictionary of input data, i.e. {'A': TensorA, 'B': TensorB}
        constants (dict[str, float]): dictionary of constants that will be used in the calculation process, i.e. {'pi': 3.14}
        outs (dict[str, tensor]): dictionary of the model's raw output, i.e. {'U': TensorU}

    Returns:
        dict[str, tensor]: dictionary of output after conditional hard enforcement

    """

    dic = {}
    for solution in solutions:
        lhs, rhs = solution
        # calculate the condition-constrained outputs through solutions
        if "outs" not in lhs:
            # Intermediate products
            dic[lhs[5:-2]] = eval_with_vars(rhs, inputs, constants, outs, dic)
        else:
            outs[lhs[6:-2]] = eval_with_vars(rhs, inputs, constants, outs, dic)
    return outs
