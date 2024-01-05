#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
from oneflow import autograd
from oneflow.utils.data import Dataset, DataLoader

if flow.cuda.is_available():
    device = "cuda"
    flow.set_default_tensor_type(flow.cuda.FloatTensor)
else:
    device = "cpu"

Module = flow.nn.Module
Tensor = flow.Tensor


def is_tensor(obj):
    return flow.is_tensor(obj)


def zeros(shape, dtype=None):
    return flow.zeros(shape, dtype=dtype, requires_grad=True).to(device)


def zeros_like(tensor):
    return flow.zeros_like(tensor, requires_grad=True).to(device)


def cat(tensor_list, dim=0):
    return flow.cat(tensor_list, dim=dim)


def unsqueeze(tensor, dim):
    return flow.unsqueeze(tensor, dim)


def squeeze(tensor, dim):
    return flow.squeeze(tensor, dim)


def forward(model, x):
    return model(x)


def np_to_tensor(array, requires_grad=False):
    return flow.tensor(array, requires_grad=requires_grad, dtype=flow.float32).to(device)


def sin(tensor):
    return flow.sin(tensor)


def cos(tensor):
    return flow.cos(tensor)


def tan(tensor):
    return flow.tan(tensor)


def arcsin(tensor):
    return flow.arcsin(tensor)


def arccos(tensor):
    return flow.arccos(tensor)


def arctan(tensor):
    return flow.arctan(tensor)


def sinh(tensor):
    return flow.sin(tensor)


def cosh(tensor):
    return flow.cos(tensor)


def tanh(tensor):
    return flow.tan(tensor)


def arcsinh(tensor):
    return flow.sin(tensor)


def arccosh(tensor):
    return flow.cos(tensor)


def arctanh(tensor):
    return flow.tan(tensor)


def power(tensor, exponent):
    return flow.pow(tensor, exponent=exponent)


def exp(tensor):
    return flow.exp(tensor)


def log(tensor):
    return flow.log(tensor)


def log2(tensor):
    return flow.log2(tensor)


def log10(tensor):
    return flow.log10(tensor)


def sqrt(tensor):
    return flow.sqrt(tensor)


def grad(y, x):
    """
    Calculate dy/dx.

    Args:
        y (tensor):
        x (tensor):

    Returns:
        dy/dx
    """
    return autograd.grad(y, x, grad_outputs=flow.ones_like(x), create_graph=True)[0]


def save(model, path):
    flow.save(model, path)


def load(path):
    return flow.load(path)
