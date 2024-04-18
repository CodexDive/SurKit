#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import time
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument('--backend', type=str, default="pytorch")
parser.add_argument('--cuda', type=str, default="3")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# choose backend from pytorch, oneflow, jax
os.environ['SRK_BACKEND'] = args.backend


import surkit.backend as bkd
from surkit.nn import fnn
from surkit.train import train

layers = [40] * 6
activation = "Tanh"
initializer = "xavier normal"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-3
x = {"low": -2.5, "high": 2.5, "size": (1000, 1)}
y = {"low": -2.5, "high": 2.5, "size": (1000, 1)}
t = {"low": 0.0, "high": 1.0, "size": (1000, 1)}
input = {"x": x, "y": y, "t": t}
output = ["h", "u", "v"]
constant = {"g": 1.0, "radius": 0.5195254015709299}
max_iteration = 10000
save_path = 'model/swe2d_%s_1' % bkd.backend_name


pde = [
    "D[h, t] + D[h, x] * u + h * D[u, x] + D[h, y] * v + h * D[v, y] = 0",
    "D[u, t] + u * D[u, x] + v * D[u, y] + g * D[h, x] = 0",
    "D[v, t] + u * D[v, x] + v * D[v, y] + g * D[h, y] = 0",
]
icbc = [
    "D[h, x] = 0 | x = -2.5",
    "D[h, x] = 0 | x = 2.5",
    "D[h, y] = 0 | y = -2.5",
    "D[h, y] = 0 | y = 2.5",
    "h = 2.0 * (Sqrt[x ** 2 + y ** 2] < radius) + 1.0 * (Sqrt[x ** 2 + y ** 2] > radius)  | t = 0",
    "u = 0 | t = 0",
    "v = 0 | t = 0",
]


net = fnn.FNN(layers, activation, in_d=3, out_d=3, initializer=initializer)

train(input=input, output=output, target=None, constant=constant, icbc=icbc, pde=pde, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=1000)