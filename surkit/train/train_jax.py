#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import pickle
# from oneflow.utils.data import DataLoader
import re
import warnings
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state

from .. import losses
from ..data.sampling import random_sampler
from ..losses.losses_jax import elbo
from ..optimizer import optimizer_jax as optim
from ..pde.pde import parse_pde
from ..utils.parse import eval_with_vars, split_pde, split_condition, split_hard_condition, loss_from_list, \
    loss_from_dict


def train(
        input: dict,
        output: list,
        net: nn.Module,
        iterations: int,
        evaluation: dict = None,
        path: str = None,
        report_interval: int = 1000,
        optimizer: str = "Adam",
        loss_function: str = "MSE",
        lr: float = 1e-5,
        target: dict = None,
        constant: dict = None,
        icbc: list = None,
        pde: list = None,
        weight: list = None,
        hard_condition: list = None,
        pde_coef: float = 1.,
        condition_coef: float = 1.,
        gt_coef: float = 1.,
):
    """
    Training step for FNN, PFNN.

    Args:
        input (dict[str, np.array] | dict[str, dict[str, int | tuple(int, int)]]): input dict, {variable name: variable value} or {variable name: variable sample range}
        output (list): list of output names
        net (nn.Module): network model
        iterations (int): max iterations
        path (str, optional): model save path
        report_interval (int, optional): how many epochs for an evaluation and report
        optimizer (str, optional): optimizer name
        loss_function (str, optional): loss_function name
        lr (float, optional): learning rate
        target (dict, optional): output dict, {output name: output value}
        constant (dict, optional): constant dict, {constant name: constant value}
        icbc (list, optional): list of initial condition and boundary condition
        pde (list, optional): list of pde
        weight (list, optional): weights for each set of inputs
        hard_condition (list, optional): hard condition of icbc which given by expert knowledge
        pde_coef (float, optional): weight of pde loss
        condition_coef (float, optional): weight of icbc loss
        gt_coef (float, optional): weight of ground truth loss
    """
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn("The save path is not specified, the model will not be saved")

    loss_func = losses.get(loss_function) if not weight else losses.get(loss_function, 'none')
    # net
    optimizer = optim.get(optimizer)
    optimizer = optimizer(learning_rate=lr)

    init_key = jax.random.PRNGKey(0)
    params = net.init(init_key, jax.random.uniform(init_key, (1, len(input))))
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)
    # print(state.params)
    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    pde_intermediate_dict = {}
    icbc_intermediate_dict = {}
    # no sampling required
    for key, value in input.items():
        if isinstance(value, np.ndarray):
            input_tensor_dict[key] = jax.device_put(value)
            if evaluation: eval_input_tensor_dict[key] = jax.device_put(evaluation[key])
    if target:
        for key, value in target.items():
            if isinstance(value, np.ndarray):
                ground_truth[key] = jax.device_put(value)
                if evaluation: eval_ground_truth[key] = jax.device_put(evaluation[key])

    def fwd(params_, x):
        return state.apply_fn(params_, x)

    out_dic = {o : None for o in output}
    grad_fn_dict = {}

    def jax_grad(variable: str):
        if variable not in grad_fn_dict:
            left, _, right = variable.rpartition('_')
            ind_x = list(input_tensor_dict.keys()).index(right)
            if '_' not in left:
                grad_fn_dict[variable] = jax.grad(
                    lambda params_, x_: jnp.sum(fwd(params_, x_)[:, output.index(left)]), 1), ind_x
            else:
                if len(input_tensor_dict) > 1:
                    grad_fn_dict[variable] = jax.grad(
                        lambda params_, x_: jnp.sum(
                            grad_fn_dict[left][0](params_, x_)[:, ind_x]), 1), ind_x
                else:
                    grad_fn_dict[variable] = jax.grad(
                        lambda params_, x_: jnp.sum(grad_fn_dict[left][0](params_, x_)), 1), ind_x

    def parse_icbc(condition, params_, x_):
        out_dic_ = {}
        if len(condition) == 3:
            # for periodic BC
            cond, when_left, when_right = condition
            left, right = cond.split('=')
            input_tensor_dict_copy_l, out_dic_l, icbc_intermediate_dict_l = out_with_cond(out_dic_, params_, [when_left.split('=')])
            ll = eval_with_vars(left, input_tensor_dict_copy_l, constant, out_dic_l, icbc_intermediate_dict_l)
            input_tensor_dict_copy_r, out_dic_r, icbc_intermediate_dict_r = out_with_cond(out_dic_, params_, [when_right.split('=')])
            rr = eval_with_vars(right, input_tensor_dict_copy_r, constant, out_dic_r, icbc_intermediate_dict_r)
            return [(ll, rr)]
        else:
            conds, whens = condition
            input_tensor_dict_copy, out_dic_, icbc_intermediate_dict = out_with_cond(out_dic_, params_, whens)
            value_target_pairs = []
            for cond in conds:
                value_target_pairs.append(
                    parse_pde(cond, input_tensor_dict_copy, constant, out_dic_, icbc_intermediate_dict)
                )
            return value_target_pairs

    def out_with_cond(out_dic_, params_, whens):
        input_tensor_dict_copy = input_tensor_dict.copy()
        for when in whens:
            variable, value = when
            variable = re.search(r"\['(.*?)'\]", variable).group(1)
            input_tensor_dict_copy[variable] = jnp.zeros_like(input_tensor_dict_copy[variable]) + eval_with_vars(value,
                                                                                                                 input_tensor_dict,
                                                                                                                 constant,
                                                                                                                 {}, {})
        out_ = fwd(params_, jnp.concatenate(list(input_tensor_dict_copy.values()), axis=1))
        for index, out in enumerate(output):
            out_dic_[out] = jnp.expand_dims(out_[:, index], axis=1)
        for key_ in icbc_intermediate_dict.keys():
            grad_fn, ind = grad_fn_dict[key_]
            if len(input_tensor_dict) > 1:
                icbc_intermediate_dict[key_] = grad_fn(params_,
                                                       jnp.concatenate(list(input_tensor_dict_copy.values()), axis=1))[
                                               :, ind].reshape(-1, 1)
            else:
                icbc_intermediate_dict[key_] = grad_fn(params_,
                                                       jnp.concatenate(list(input_tensor_dict_copy.values()), axis=1))
        return input_tensor_dict_copy, out_dic_, icbc_intermediate_dict

    def loss_fn(params_, x):
        out_ = fwd(params_, jnp.concatenate(x, axis=1))
        for index, out in enumerate(output):
            out_dic[out] = jnp.expand_dims(out_[:, index], axis=1)

        # pde
        for key_ in pde_intermediate_dict.keys():
            grad_fn, ind = grad_fn_dict[key_]
            if len(input_tensor_dict) > 1:
                pde_intermediate_dict[key_] = grad_fn(params_, jnp.concatenate(x, axis=1))[:, ind].reshape(-1, 1)
            else:
                pde_intermediate_dict[key_] = grad_fn(params_, jnp.concatenate(x, axis=1))
        pde_losses = [parse_pde(equation, input_tensor_dict, constant, out_dic, pde_intermediate_dict) for equation in pde]
        pde_loss = loss_from_list(loss_func, pde_losses, weight)

        icbc_losses = []
        for condition in icbc:
            icbc_losses += parse_icbc(condition, params_, x)
        icbc_loss = loss_from_list(loss_func, icbc_losses, weight)

        gt_loss = loss_from_dict(loss_func, out_dic, ground_truth, weight)

        return pde_coef * pde_loss + condition_coef * icbc_loss + gt_coef * gt_loss

    def train_step(state_):
        loss_, grads = (jax.value_and_grad(loss_fn))(state_.params, list(input_tensor_dict.values()))
        new_state = state_.apply_gradients(grads=grads)
        return new_state, loss_

    jit_train_step = jax.jit(train_step)
    # jit_train_step = train_step
    for epoch in range(iterations):
        # sampling
        for key, value in input.items():
            if isinstance(value, dict):
                input_tensor_dict[key] = jnp.array(random_sampler(value["low"], value["high"], value["size"]))
        if epoch == 0:
            if pde:
                for i, v in enumerate(pde):
                    pde[i] = split_pde(v, input_tensor_dict, constant, out_dic, pde_intermediate_dict)
            if icbc:
                for i, v in enumerate(icbc):
                    icbc[i] = split_condition(v, input_tensor_dict, constant, out_dic, icbc_intermediate_dict)
            if hard_condition:
                for i, v in enumerate(hard_condition):
                    hard_condition[i] = split_hard_condition(v, input_tensor_dict, constant, out_dic, {})

            for key in pde_intermediate_dict.keys():
                jax_grad(key)

            for key in icbc_intermediate_dict.keys():
                jax_grad(key)

        # if hard_condition:
        #     out_dic = convert(hard_condition, input_tensor_dict, constant, out_dic)

        state, loss = jit_train_step(state)
        # report loss
        if (epoch + 1) % report_interval == 0:
            # save parameters
            if evaluation:
                eval_out_dic = {}
                predict = state.apply_fn(state.params, jnp.concatenate(list(eval_input_tensor_dict.values()), axis=1))
                for index, out in enumerate(output):
                    eval_out_dic[out] = jnp.expand_dims(predict[:, index], axis=1)
                eval_loss = loss_from_dict(loss_func, eval_out_dic, eval_ground_truth, weight)
                print(epoch + 1, "Train Loss:", loss, "Eval Loss:", eval_loss)
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss
                    state_dict = flax.serialization.to_state_dict(state)
                    pickle.dump(state_dict, open(path, "wb"))
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss)
                if path:
                    if loss < best_loss:
                        best_loss = loss
                        state_dict = flax.serialization.to_state_dict(state)
                        pickle.dump(state_dict, open(path, "wb"))
                        print("Epoch", epoch + 1, "saved", path)

def train_gaussian(
        input: dict,
        output: list,
        net,
#: nn.Module,
        iterations: int,
        evaluation: dict = None,
        path: str = None,
        report_interval: int = 1,
        optimizer: str = "Adam",
        lr: float = 1e-5,
        target: dict = None,
):
    """
    Training step for NN-ensemble.

    Args:
        input (dict[str, np.array]): input dict, {variable name: variable value}
        output (list): list of output names
        net (nn.Module): network model
        iterations (int): max iterations
        path (str, optional): model save path
        report_interval (int, optional): how many epochs for an evaluation and report
        optimizer (str, optional): optimizer name
        lr (float, optional): learning rate
        target (dict, optional): output dict, {output name: output value}
    """
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn("The save path is not specified, the model will not be saved")

    loss_func = losses.get("gaussiannll")
    # net
    optimizer = optim.get(optimizer)
    optimizer = optimizer(learning_rate=lr)

    init_key = jax.random.PRNGKey(0)
    params = net.init(init_key, jax.random.uniform(init_key, (1, len(input))))
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)
    # print(state.params)
    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    # no sampling required
    for key, value in input.items():
        if isinstance(value, np.ndarray):
            input_tensor_dict[key] = jax.device_put(value)
            if evaluation: eval_input_tensor_dict[key] = jax.device_put(evaluation[key])
    if target:
        for key, value in target.items():
            if isinstance(value, np.ndarray):
                ground_truth[key] = jax.device_put(value)
                if evaluation: eval_ground_truth[key] = jax.device_put(evaluation[key])

    def fwd(params_, x):
        return state.apply_fn(params_, x, True)

    def loss_fn(params_, x):
        means, variances = fwd(params_, jnp.concatenate(x, axis=1))
        gt = jnp.concatenate(list(ground_truth.values()), axis=1)
        gt_loss = jnp.array([loss_func(gt, m, v) for m, v in zip(means, variances)]).mean()
        return gt_loss

    def train_step(state_):
        loss_, grads = (jax.value_and_grad(loss_fn))(state_.params, list(input_tensor_dict.values()))
        new_state = state_.apply_gradients(grads=grads)
        return new_state, loss_

    jit_train_step = jax.jit(train_step)
    # jit_train_step = train_step
    for epoch in range(iterations):
        state, loss = jit_train_step(state)
        # report loss
        if (epoch + 1) % report_interval == 0:
            if evaluation:
                eval_gt = jnp.concatenate(list(eval_ground_truth.values()), axis=1)
                predict_mean, _ = state.apply_fn(state.params, jnp.concatenate(list(eval_input_tensor_dict.values()), axis=1))
                eval_loss = losses.get("mse")(predict_mean, eval_gt)
                print(epoch + 1, "Train Loss:", loss, "Eval Loss (l2):", eval_loss)
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss
                    state_dict = flax.serialization.to_state_dict(state)
                    pickle.dump(state_dict, open(path, "wb"))
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss)
                # save parameters
                if path:
                    if loss < best_loss:
                        best_loss = loss
                        state_dict = flax.serialization.to_state_dict(state)
                        pickle.dump(state_dict, open(path, "wb"))
                        print("Epoch", epoch + 1, "saved", path)

def train_bayesian(
        input: dict,
        output: list,
        net: nn.Module,
        iterations: int,
        evaluation: dict = None,
        path: str = None,
        report_interval: int = 1,
        optimizer: str = "Adam",
        lr: float = 1e-5,
        target: dict = None,
        samples: int = 32,
):
    """
    Training step for BNN.

    Args:
        input (dict[str, np.array]): input dict, {variable name: variable value}
        output (list): list of output names
        net (nn.Module): network model
        iterations (int): max iterations
        path (str, optional): model save path
        report_interval (int, optional): how many epochs for an evaluation and report
        optimizer (str, optional): optimizer name
        lr (float, optional): learning rate
        target (dict, optional): output dict, {output name: output value}
        samples (int, optional): how many times the parameters need to be sampled for a prediction

    """
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn("The save path is not specified, the model will not be saved")

    # net
    optimizer = optim.get(optimizer)
    optimizer = optimizer(learning_rate=lr)

    init_key = jax.random.PRNGKey(0)
    params = net.init(init_key, jax.random.uniform(init_key, (1, len(input))))
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)
    # print(state.params)
    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    # no sampling required
    for key, value in input.items():
        if isinstance(value, np.ndarray):
            input_tensor_dict[key] = jax.device_put(value)
            if evaluation: eval_input_tensor_dict[key] = jax.device_put(evaluation[key])
    if target:
        for key, value in target.items():
            if isinstance(value, np.ndarray):
                ground_truth[key] = jax.device_put(value)
                if evaluation: eval_ground_truth[key] = jax.device_put(evaluation[key])



    def loss_fn(params_):
        return elbo(jnp.concatenate(list(input_tensor_dict.values()), axis=1),
                    jnp.concatenate(list(ground_truth.values()), axis=1), samples, net, params_)

    def train_step(state_):
        loss_, grads = (jax.value_and_grad(loss_fn))(state_.params)
        new_state = state_.apply_gradients(grads=grads)
        return new_state, loss_

    jit_train_step = jax.jit(train_step)
    for epoch in range(iterations):
        state, loss = jit_train_step(state)
        # report loss
        if (epoch + 1) % report_interval == 0:
            if evaluation:
                eval_gt = jnp.concatenate(list(eval_ground_truth.values()), axis=1)
                predict = np.mean([state.apply_fn(state.params, jnp.concatenate(list(eval_input_tensor_dict.values()), axis=1))
                               for _ in range(samples)], axis=0)
                eval_loss = losses.get("mse")(predict, eval_gt)
                print(epoch + 1, "Train Loss:", loss, "Eval Loss (l2):", eval_loss)
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss
                    state_dict = flax.serialization.to_state_dict(state)
                    pickle.dump(state_dict, open(path, "wb"))
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss)
                # save parameters
                if path:
                    if loss < best_loss:
                        best_loss = loss
                        state_dict = flax.serialization.to_state_dict(state)
                        pickle.dump(state_dict, open(path, "wb"))
                        print("Epoch", epoch + 1, "saved", path)