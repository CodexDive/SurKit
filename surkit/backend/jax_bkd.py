#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import pickle

import flax
# import flax.training.checkpoints
import flax.linen as nn
import flax.training.train_state
import jax
import jax.numpy as jnp

Module = nn.Module
Tensor = jnp.ndarray

def is_tensor(obj):
    return isinstance(obj, jnp.ndarray)


def zeros(shape, dtype=None):
    return jnp.zeros(shape, dtype=dtype)


def zeros_like(tensor):
    return jnp.zeros_like(tensor)


def cat(tensor_list, dim=0):
    return jnp.concatenate(tensor_list, axis=dim)


def unsqueeze(tensor, dim):
    return jnp.expand_dims(tensor, dim)


def squeeze(tensor, dim):
    return jnp.squeeze(tensor, dim)


@jax.jit
def forward(model, x, params=None):
    if type(model) == flax.training.train_state.TrainState:
        return model.apply_fn(model.params, x)
    return model.apply(params, x)


def np_to_tensor(array):
    return jnp.array(array)
    # return jax.device_put(array)


@jax.jit
def sin(tensor):
    return jnp.sin(tensor)


@jax.jit
def cos(tensor):
    return jnp.cos(tensor)


@jax.jit
def tan(tensor):
    return jnp.tan(tensor)


@jax.jit
def arcsin(tensor):
    return jnp.arcsin(tensor)


@jax.jit
def arccos(tensor):
    return jnp.arccos(tensor)


@jax.jit
def arctan(tensor):
    return jnp.arctan(tensor)


@jax.jit
def sinh(tensor):
    return jnp.sinh(tensor)


@jax.jit
def cosh(tensor):
    return jnp.cosh(tensor)


@jax.jit
def tanh(tensor):
    return jnp.tanh(tensor)


@jax.jit
def arcsinh(tensor):
    return jnp.arcsinh(tensor)


@jax.jit
def arccosh(tensor):
    return jnp.arccosh(tensor)


@jax.jit
def arctanh(tensor):
    return jnp.arctanh(tensor)


@jax.jit
def power(tensor, exponent):
    return jnp.power(tensor, exponent)


@jax.jit
def exp(tensor):
    return jnp.exp(tensor)


@jax.jit
def log(tensor):
    return jnp.log(tensor)


@jax.jit
def log2(tensor):
    return jnp.log2(tensor)


@jax.jit
def log10(tensor):
    return jnp.log10(tensor)


@jax.jit
def sqrt(tensor):
    return jnp.sqrt(tensor)


@jax.jit
def grad(state, x, ind_x, ind_y):
    """
    Calculate dy/dx.

    Args:
        y (tensor):
        x (tensor):

    Returns:
        dy/dx
    """
    return jax.grad(lambda x : jnp.sum(state.apply_fn(state.params, x)[:, ind_y]), 1)(x)[:, ind_x]


def save(state, path):
    state_dict=flax.serialization.to_state_dict(state)
    pickle.dump(state_dict,open(path,"wb"))


def load(state, path):
    pkl_file = pickle.load(open(path, "rb"))
    state = flax.serialization.from_state_dict(target=state, state=pkl_file)
    return state
