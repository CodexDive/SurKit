#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np

from .. import backend as bkd


def exp(x):
    if bkd.is_tensor(x):
        return bkd.exp(x)
    else:
        return np.exp(x)

def sin(x):
    if bkd.is_tensor(x):
        return bkd.sin(x)
    else:
        return np.sin(x)

def cos(x):
    if bkd.is_tensor(x):
        return bkd.cos(x)
    else:
        return np.cos(x)

def tan(x):
    if bkd.is_tensor(x):
        return bkd.tan(x)
    else:
        return np.tan(x)

def arcsin(x):
    if bkd.is_tensor(x):
        return bkd.arcsin(x)
    else:
        return np.arcsin(x)

def arccos(x):
    if bkd.is_tensor(x):
        return bkd.arccos(x)
    else:
        return np.arccos(x)

def arctan(x):
    if bkd.is_tensor(x):
        return bkd.arctan(x)
    else:
        return np.arctan(x)

def sinh(x):
    if bkd.is_tensor(x):
        return bkd.sinh(x)
    else:
        return np.sinh(x)

def cosh(x):
    if bkd.is_tensor(x):
        return bkd.cosh(x)
    else:
        return np.cosh(x)

def tanh(x):
    if bkd.is_tensor(x):
        return bkd.tanh(x)
    else:
        return np.tanh(x)

def arcsinh(x):
    if bkd.is_tensor(x):
        return bkd.arcsinh(x)
    else:
        return np.arcsinh(x)

def arccosh(x):
    if bkd.is_tensor(x):
        return bkd.arccosh(x)
    else:
        return np.arccosh(x)

def arctanh(x):
    if bkd.is_tensor(x):
        return bkd.arctanh(x)
    else:
        return np.arctanh(x)

def log(x, y=None):
    if bkd.is_tensor(x):
        if y:
            return bkd.log(y) / bkd.log(x)
        return bkd.log(x)
    else:
        if y:
            return np.log(y) / np.log(x)
        return np.log(x)

def log2(x):
    if bkd.is_tensor(x):
        return bkd.log2(x)
    else:
        return np.log2(x)

def log10(x):
    if bkd.is_tensor(x):
        return bkd.log10(x)
    else:
        return np.log10(x)

def sqrt(x):
    if bkd.is_tensor(x):
        return bkd.sqrt(x)
    else:
        return np.sqrt(x)

pi = np.pi
e = np.exp(1)

Exp = exp
Sin = sin
Cos = cos
Tan = tan
ArcSin = arcsin
ArcCos = arccos
ArcTan = arctan
Sinh = sinh
Cosh = cosh
Tanh = tanh
ArcSinh = arcsinh
ArcCosh = arccosh
ArcTanh = arctanh
Log = log
Log2 = log2
Log10 = log10
Sqrt = sqrt

Pi = pi
E = e

math_list = [
    "Exp",
    "exp",
    "Sin",
    "sin",
    "Cos",
    "cos",
    "Tan",
    "tan",
    "ArcSin",
    "arcsin",
    "ArcCos",
    "arccos",
    "ArcTan",
    "arctan",
    "Sinh",
    "sinh",
    "Cosh",
    "cosh",
    "Tanh",
    "tanh",
    "ArcSinh",
    "arcsinh",
    "ArcCosh",
    "arccosh",
    "ArcTanh",
    "arctanh",
    "Log",
    "log",
    "Log2",
    "log2",
    "Log10",
    "log10",
    "Sqrt",
    "sqrt",
    "Pi",
    "pi",
    "E",
    "e",
]