#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import argparse
import re

from .math_utils import *


def parse_dict(s):
    try:
        d = eval(s)
        if not isinstance(d, dict):
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: '{s}'")
        return d
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: '{s}'")

def safe_compile(code):
    """
    The eval function that filters out harmful strings

    Args:
        code (str): A string of python code that can be run by eval

    Returns:
        code: a complied string
    """
    banned = ('eval', 'compile', 'exec', 'getattr', 'hasattr', 'setattr', 'delattr',
              'classmethod', 'globals', 'help', 'input', 'isinstance', 'issubclass', 'locals',
              'open', 'print', 'property', 'staticmethod', 'vars')

    # Check for banned names in the code
    for name in re.findall(r'\b\w+\b', code):
        if name in banned:
            raise NameError(f'{name} not allowed : arbitrary code execution not allowed')

    # Compile and evaluate the code
    return compile(code, '', 'eval')

def eval_with_vars(code, inputs, constants, outs, dic):
    """
    Run a code string with dictionaries as vars.

    Args:
        code (str): code to be executed
        inputs (dict[str, Tensor]): dict with input features' names and their values
        constants (dict[str, int | float]): dict with constants' names and their values
        outs (dict[str, Tensor]): dict with output features' names and their values
        dic (dict[str, Tensor]): dict with gradients or intermediate variables' names and their values

    Returns:
        Any: result of the code string
    """
    return eval(code)

def get_grad(variable: str, inputs, dic, outs):
    """
    Get grad through backend's automatic differentiation.

    Args:
        variable (str): code to be executed
        inputs (dict[str, Tensor]): dict with input features' names and their values
        dic (dict[str, int | float]): dict with calculated gradients' names and their values
        outs (dict[str, Tensor]): dict with output features' names and their values

    Returns:
        Any: result of the code string
    """
    y, _, x = variable.rpartition('_')

    return bkd.grad(outs[y], inputs[x]) if '_' not in y else bkd.grad(dic[y], inputs[x])

def get_grad2d(variable: str, inputs, dic, outs, imgfilter=None):
    y, x = variable.split('_')
    if ".h" in variable:
        return imgfilter.grad_h(outs[y])
    elif ".v" in variable:
        return imgfilter.grad_v(outs[y])
    else:
        pass

def convert_math_to_python(equation: str):
    def repl(match):
        return match.group(1) + ("_" + match.group(1)) * (int(match.group(2)) - 1)

    equation = equation.replace('[', '(')
    equation = equation.replace(']', ')')
    equation = equation.replace('^', '**')

    pattern = r"\s*{(.*?),\s*(\d+)}"
    while re.search(pattern, equation):
        equation = re.sub(pattern, repl, equation)

    pattern = r"D\((.*?),\s*(.*?)\)"
    while re.search(pattern, equation):
        equation = re.sub(pattern, r"\1_\2", equation)

    return equation

def equation_to_runnable(equation: str, inputs, constants, outs, dic):
    """
    Parse the Mathematica-style formula string into python-executable code strings by substituting the variable name string with the “dict[variable name]” string.
    i.e. "D[y, x] + x = g * y" => "grads['y_x'] + inputs['x'] = constants['g'] * outs['y']",
     both lhs and rhs of the equation are python-executable code, which are convenient for subsequent calculations
    Args:
        equation (str): a string contains user defined variable
        inputs (dict[str, Tensor]): dict with input features' names and their values
        constants (dict[str, int | float]): dict with constants' names and their values
        outs (dict[str, Tensor]): dict with output features' names and their values, values can be None
        dic (dict[str, Tensor]): dict with gradients or intermediate variables' names and their values, values can be None
        dic (dict[str, Tensor]): dict with gradients or intermediate variables' names and their values, values can be None
    Returns:
        str: an executable code string
    """
    equation = convert_math_to_python(equation)
    equation = re.sub(r"\s+", "", equation)
    variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', equation)

    original_pattern = lambda variable : r'\b' + variable + r'\b'
    replacement_pattern = {
        "inputs" : lambda variable : "inputs['%s']" % variable,
        "constants" : lambda variable : "constants['%s']" % variable,
        "outs" : lambda variable : "outs['%s']" % variable,
        "dic" : lambda variable : "dic['%s']" % variable,
    }
    replace = lambda variable: re.sub(original_pattern(variable), locate_var_in_dicts(variable), equation)

    def locate_var_in_dicts(variable: str):
        if variable in inputs:
            return replacement_pattern['inputs'](variable)
        if variable in constants:
            return replacement_pattern['constants'](variable)
        if variable in outs:
            return replacement_pattern['outs'](variable)
        if '_' in variable:
            parts = variable.split('_')
            y_x = parts[0] + '_' + parts[1]
            dic[y_x] = None
            for x in parts[2:]:
                y_x += '_' + x
                dic[y_x] = None
        elif '|' in variable:
            dic[variable] = None
        return replacement_pattern['dic'](variable)

    for variable in set(variables):
        if variable not in math_list:
            equation = replace(variable)
    return equation

def split_pde(equation: str, inputs, constants, outs, dic):
    code = equation_to_runnable(equation, inputs, constants, outs, dic)
    left, right = code.split('=')
    return [safe_compile(left), safe_compile(right)]

def split_condition(equation: str, inputs, constants, outs, dic):
    if '|' not in equation:
        # for periodic BC
        equation = re.sub(r"\s+", "", equation)
        matches = re.match(r"(.*\[(.*?)=(.*?)\].*=.*\[(.*?)=(.*?)\].*)", equation)
        cond = equation_to_runnable(
            matches.group(1).replace("[" + matches.group(2) + "=" + matches.group(3) + "]", "").replace(
                "[" + matches.group(4) + "=" + matches.group(5) + "]", ""), inputs, constants, outs, dic)
        when_left = equation_to_runnable(matches.group(2) + "=" + matches.group(3), inputs, constants, outs, dic)
        when_right = equation_to_runnable(matches.group(4) + "=" + matches.group(5), inputs, constants, outs, dic)
        return [cond, when_left, when_right]

    code = equation_to_runnable(equation, inputs, constants, outs, dic)
    conds, whens = code.split('|')
    cond_list = []
    when_list = []
    for cond in conds.split(','):
        left, right = cond.split('=')
        cond_list.append([safe_compile(left), safe_compile(right)])

    for when in whens.split(','):
        when_left, when_right = when.split('=')
        when_list.append([when_left, safe_compile(when_right)])
    return [cond_list, when_list]

def split_hard_condition(equation: str, inputs, constants, outs, dic):
    code = equation_to_runnable(equation, inputs, constants, outs, dic)
    left, right = code.split('=')
    return [left, safe_compile(right)]

def loss_from_list(loss_func, data, weight):
    if weight:
        return sum((loss_func(pair[0], pair[1]) * weight).mean() for pair in data)
    return sum(loss_func(pair[0], pair[1]) for pair in data)

def loss_from_dict(loss_func, predict_dict, ground_truth_dict, weight):
    if weight:
        return sum((loss_func(value, predict_dict[key]) * weight).mean() for key, value in ground_truth_dict.items())
    return sum(loss_func(value, predict_dict[key]) for key, value in ground_truth_dict.items())