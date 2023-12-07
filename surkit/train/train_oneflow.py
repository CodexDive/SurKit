#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import warnings
from pathlib import Path

import numpy as np
import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader

from .. import losses
from ..conditions.hard_condition import convert
from ..conditions.icbc import parse_icbc
from ..conditions.icbc2d import parse_icbc2d
from ..data.processing import np_to_tensor
from ..data.sampling import random_sampler
from ..gradient.sobelfilter import SobelFilter
from ..losses.losses_oneflow import elbo
from ..optimizer import optimizer_oneflow as optim
from ..pde.pde import parse_pde
from ..utils.parse import get_grad, get_grad2d, split_pde, split_condition, split_hard_condition, loss_from_list, \
    loss_from_dict

if flow.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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
        input (dict[str, np.array] | dict[str, dict[str, int | tuple(int, int)]]): input dict, {variable name: variable value} or {variable name: variable sample range} or {variable name: variable sample range}
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

    net.to(device)
    net.train()

    loss_func = losses.get(loss_function) if not weight else losses.get(loss_function, 'none')
    # net
    optimizer = optim.get(optimizer)
    optimizer = optimizer(net.parameters(), lr=lr)
    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    pde_intermediate_dict = {}
    icbc_intermediate_dict = {}
    # 无需采样
    for key, value in input.items():
        if isinstance(value, np.ndarray):
            input_tensor_dict[key] = np_to_tensor(value)
            if evaluation: eval_input_tensor_dict[key] = np_to_tensor(evaluation[key])
    if target:
        for key, value in target.items():
            if isinstance(value, np.ndarray):
                ground_truth[key] = np_to_tensor(value)
                if evaluation: eval_ground_truth[key] = np_to_tensor(evaluation[key])

    # 训练迭代
    for epoch in range(iterations):
        optimizer.zero_grad()
        # 随机采样
        for key, value in input.items():
            if isinstance(value, dict):
                input_tensor_dict[key] = np_to_tensor(random_sampler(value["low"], value["high"], value["size"]))
        # 输出
        out_ = net(flow.cat(list(input_tensor_dict.values()), dim=1))
        out_dic = {}
        for index, out in enumerate(output):
            out_dic[out] = flow.unsqueeze(out_[:, index], dim=1)
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

        # 边界条件硬执行
        if hard_condition:
            out_dic = convert(hard_condition, input_tensor_dict, constant, out_dic)

        for key in pde_intermediate_dict.keys():
            pde_intermediate_dict[key] = get_grad(key, input_tensor_dict, pde_intermediate_dict, out_dic)
        pde_losses = [parse_pde(equation, input_tensor_dict, constant, out_dic, pde_intermediate_dict) for equation in pde]
        pde_loss = loss_from_list(loss_func, pde_losses, weight)

        icbc_losses = []
        for condition in icbc:
            icbc_losses += parse_icbc(condition, input_tensor_dict, constant, output, icbc_intermediate_dict, net)
        icbc_loss = loss_from_list(loss_func, icbc_losses, weight)

        gt_loss = loss_from_dict(loss_func, out_dic, ground_truth, weight)

        loss = pde_coef * pde_loss + condition_coef * icbc_loss + gt_coef * gt_loss
        loss.backward()
        optimizer.step()
        # 报告loss
        if (epoch + 1) % report_interval == 0:
            if evaluation:
                eval_out_dic = {}
                predict = net(flow.cat(list(eval_input_tensor_dict.values()), dim=1))
                for index, out in enumerate(output):
                    eval_out_dic[out] = flow.unsqueeze(predict[:, index], dim=1)
                eval_loss = loss_from_dict(loss_func, eval_out_dic, eval_ground_truth, weight)
                print(epoch + 1, "Train Loss:", loss.item(), "Eval Loss:", eval_loss.item())
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss.item()
                    flow.save(net, path)
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss.item())
                # 保存网络
                if path:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        flow.save(net, path)
                        print("Epoch", epoch + 1, "saved", path)


def train_gaussian(
        input: dict,
        output: list,
        net: nn.Module,
        iterations: int,
        evaluation: dict = None,
        path: str = None,
        report_interval: int = 1,
        optimizer: str = "Adam",
        lr: float = 1e-5,
        eps: float = 1e-5,
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

    net.to(device)
    net.train()
    # loss_fnc = nn.GaussianNLLLoss()
    loss_fnc = losses.get("gaussiannll")
    optimizer = optim.get(optimizer)
    optimizer_list = []
    for model in net.models:
        optimizer_list.append(optimizer(model.parameters(), lr=lr, eps=eps))

    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    # 无需采样
    for key, value in input.items():
        input_tensor_dict[key] = np_to_tensor(value)
        if evaluation: eval_input_tensor_dict[key] = np_to_tensor(evaluation[key])
    if target:
        for key, value in target.items():
            ground_truth[key] = np_to_tensor(value)
            if evaluation: eval_ground_truth[key] = np_to_tensor(evaluation[key])
    x = flow.cat(list(input_tensor_dict.values()), dim=1)
    y = flow.cat(list(ground_truth.values()), dim=1)
    # 训练迭代
    for epoch in range(iterations):
        loss_list = []
        for i in range(net.n_models):
            optimizer_list[i].zero_grad()
            pred, var = net.models[i](x)
            loss = loss_fnc(y, pred, var)
            # loss.retain_grad = True
            loss.backward(retain_graph=True)
            loss_list.append(loss.item())
            optimizer_list[i].step()

        # 报告loss
        if (epoch + 1) % report_interval == 0:
            if evaluation:
                eval_gt = flow.cat(list(eval_ground_truth.values()), dim=1)
                predict, _ = net(flow.cat(list(eval_input_tensor_dict.values()), dim=1))
                print(eval_gt)
                eval_loss = losses.get("mse")(eval_gt, predict)
                print(epoch + 1, "Train (GaussianNLLloss):", loss_list, "Eval Loss (l2):", eval_loss.item())
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss.item()
                    flow.save(net, path)
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss_list)
                # 保存网络
                if path:
                    if np.mean(loss_list) < best_loss:
                        best_loss = np.mean(loss_list)
                        flow.save(net, path)
                        print("Epoch", epoch + 1, "saved", path)


def train2d(
        input: DataLoader,
        constant: dict,
        inputs: list,
        output: list,
        icbc: list,
        pde: list,
        net: nn.Module,
        iterations: int,
        report_interval: int,
        optimizer: str,
        lr: float,
        loss_function: str,
        imsize: int,
        path: str = None,
        device='cpu',
        pde_coef: float = 1.,
        condition_coef: float = 1.,
        gt_coef: float = 1.,
):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        warnings.warn("The save path is not specified, the model will not be saved")

    net.train()
    net.to(device)
    print("<--------------training-------------->")
    # 获取损失函数，优化器配置
    loss_func = losses.get(loss_function)
    optimizer = optim.get(optimizer)
    optimizer = optimizer(net.parameters(), lr=lr)
    best_loss = float('inf')
    imgfilter = SobelFilter(imsize=imsize)
    pde_intermediate_dict = {}
    icbc_intermediate_dict = {}
    # 训练迭代
    scheduler = flow.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, div_factor=2., pct_start=0.3,
                                                   steps_per_epoch=len(input), epochs=iterations)

    for epoch in range(iterations):
        loss_sum = 0.
        for batch_idx, pair in enumerate(input):
            img = pair[0]
            optimizer.zero_grad()
            img = img.to(device)
            # 输出

            input_tensor_dict = {}
            for index, input_ in enumerate(inputs):
                input_tensor_dict[input_] = flow.unsqueeze(img[:, index], dim=1)

            out_ = net(img)
            out_dic = {}
            for index, out in enumerate(output):
                out_dic[out] = flow.unsqueeze(out_[:, index], dim=1)

            if batch_idx + epoch == 0:
                if pde:
                    for i, v in enumerate(pde):
                        pde[i] = split_pde(v, input_tensor_dict, constant, out_dic, pde_intermediate_dict)
                if icbc:
                    for i, v in enumerate(icbc):
                        icbc[i] = split_condition(v, input_tensor_dict, constant, out_dic, icbc_intermediate_dict)

            for key in pde_intermediate_dict.keys():
                pde_intermediate_dict[key] = get_grad2d(key, input_tensor_dict, pde_intermediate_dict, out_dic, imgfilter)

            for key in pde_intermediate_dict.keys():
                pde_intermediate_dict[key] = get_grad(key, input_tensor_dict, pde_intermediate_dict, out_dic)
            pde_losses = [parse_pde(equation, input_tensor_dict, constant, out_dic, pde_intermediate_dict) for equation in pde]
            pde_loss = loss_from_list(loss_func, pde_losses, None)

            icbc_losses = []
            for condition in icbc:
                icbc_losses += parse_icbc2d(condition, input_tensor_dict, constant, output, icbc_intermediate_dict)
            icbc_loss = loss_from_list(loss_func, icbc_losses, None)

            loss = pde_coef * pde_loss + condition_coef * icbc_loss
            if len(pair) == 2:
                gt_loss = gt_coef * loss_func(pair[1].to(device), out_)
                loss += gt_loss

            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 报告loss
        if (epoch + 1) % report_interval == 0:
            print(epoch + 1, "Loss:", loss_sum / (batch_idx + 1))
            # 保存网络
            if path:
                if loss_sum < best_loss:
                    best_loss = loss_sum
                    flow.save(net, path)
                    print("Epoch", epoch + 1, "saved")

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
        eps: float = 1e-5,
        target: dict = None,
        samples: int = 4,
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

    net.train()
    net.to(device)
    optimizer = optim.get(optimizer)
    optimizer = optimizer(net.parameters(), lr=lr)
    input_tensor_dict = {}
    eval_input_tensor_dict = {}
    ground_truth = {}
    eval_ground_truth = {}
    best_loss = float('inf')
    # 无需采样
    for key, value in input.items():
        input_tensor_dict[key] = np_to_tensor(value)
        if evaluation: eval_input_tensor_dict[key] = np_to_tensor(evaluation[key])
    if target:
        for key, value in target.items():
            ground_truth[key] = np_to_tensor(value)
            if evaluation: eval_ground_truth[key] = np_to_tensor(evaluation[key])
    x = flow.cat(list(input_tensor_dict.values()), dim=1)
    y = flow.cat(list(ground_truth.values()), dim=1)
    # 训练迭代
    for epoch in range(iterations):
        optimizer.zero_grad()
        loss = elbo(x, y, samples, net)
        loss.backward()
        optimizer.step()
        # 报告loss
        if (epoch + 1) % report_interval == 0:
            if evaluation:
                eval_gt = flow.cat(list(eval_ground_truth.values()), axis=1)
                predict = np.mean([net(flow.cat(list(eval_input_tensor_dict.values()), axis=1))
                     for _ in range(samples)], axis=0)
                eval_loss = losses.get("mse")(predict, eval_gt)
                print(epoch + 1, "Train Loss:", loss, "Eval Loss (l2):", eval_loss)
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss
                    flow.save(net, path)
                    print("Epoch", epoch + 1, "saved", path)
            else:
                print(epoch + 1, "Loss:", loss)
                # 保存网络
                if path:
                    if loss < best_loss:
                        best_loss = loss
                        flow.save(net, path)
                        print("Epoch", epoch + 1, "saved", path)