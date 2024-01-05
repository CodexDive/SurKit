#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import torch
from torch import autograd
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

Module = torch.nn.Module
Tensor = torch.Tensor


def is_tensor(obj):
    return torch.is_tensor(obj)


def zeros(shape, dtype=None):
    return torch.zeros(shape, dtype=dtype, requires_grad=True)


def zeros_like(tensor):
    return torch.zeros_like(tensor, requires_grad=True)


def cat(tensor_list, dim=0):
    return torch.cat(tensor_list, dim=dim)


def unsqueeze(tensor, dim):
    return torch.unsqueeze(tensor, dim)


def squeeze(tensor, dim):
    return torch.squeeze(tensor, dim)


def forward(model, x):
    return model(x)


def np_to_tensor(array, requires_grad=False):
    return torch.tensor(array, requires_grad=requires_grad, dtype=torch.float32)


def sin(tensor):
    return torch.sin(tensor)


def cos(tensor):
    return torch.cos(tensor)


def tan(tensor):
    return torch.tan(tensor)


def arcsin(tensor):
    return torch.arcsin(tensor)


def arccos(tensor):
    return torch.arccos(tensor)


def arctan(tensor):
    return torch.arctan(tensor)


def sinh(tensor):
    return torch.sinh(tensor)


def cosh(tensor):
    return torch.cosh(tensor)


def tanh(tensor):
    return torch.tanh(tensor)


def arcsinh(tensor):
    return torch.arcsinh(tensor)


def arccosh(tensor):
    return torch.arccosh(tensor)


def arctanh(tensor):
    return torch.arctanh(tensor)


def power(tensor, exponent):
    return torch.pow(tensor, exponent=exponent)


def exp(tensor):
    return torch.exp(tensor)


def log(tensor):
    return torch.log(tensor)


def log2(tensor):
    return torch.log2(tensor)


def log10(tensor):
    return torch.log10(tensor)


def sqrt(tensor):
    return torch.sqrt(tensor)


def grad(y, x):
    """
    Calculate dy/dx.

    Args:
        y (tensor):
        x (tensor):

    Returns:
        dy/dx
    """
    return autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]


def save(model, path):
    torch.save(model, path)


def load(path):
    return torch.load(path)
