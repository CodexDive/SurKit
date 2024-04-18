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


import surkit
import surkit.backend as bkd
from surkit.nn import fnn
from surkit.train import train

import h5py

data_path = "./dataset/2D_rdb_NA_NA.h5"
np.random.seed(2300)
with h5py.File(data_path, "r") as h5_file:
    seed_group = h5_file["0000"]
    x = np.array(seed_group["grid"]["x"])
    y = np.array(seed_group["grid"]["y"])
    t = np.array(seed_group["grid"]["t"])
    data = np.array(seed_group['data'])
    inp = np.array(np.meshgrid(x, y, t)).T.reshape(-1, 3)
    h = data.reshape(len(x) * len(y) * len(t), -1)
    index = np.random.choice(range(len(h) - 1), 15000)

    x_train = inp[index[:10000], 0].reshape(-1, 1)
    y_train = inp[index[:10000], 1].reshape(-1, 1)
    t_train = inp[index[:10000], 2].reshape(-1, 1)
    h_train = h[index[:10000]]

    x_eval = inp[index[10000:], 0].reshape(-1, 1)
    y_eval = inp[index[10000:], 1].reshape(-1, 1)
    t_eval = inp[index[10000:], 2].reshape(-1, 1)
    h_eval = h[index[10000:]]

layers = [40] * 6
activation = "Tanh"
initializer = "xavier normal"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-3
input = {"x": x_train, "y": y_train, "t": t_train}
output = ["h", "u", "v"]
constant = {"g": 1.0, "radius": 0.5195254015709299}
max_iteration = 10000
save_path = 'model/swe2d_with_full_data_%s' % bkd.backend_name

# Periodic BC
pde = [
    "D[h, t] + D[h, x] * u + h * D[u, x] + D[h, y] * v + h * D[v, y] = 0",
    "D[u, t] + u * D[u, x] + v * D[u, y] + g * D[h, x] = 0",
    "D[v, t] + u * D[v, x] + v * D[v, y] + g * D[h, y] = 0"
]
icbc = [
    "D[h, x] = 0 | x = -2.5",
    "D[h, x] = 0 | x = 2.5",
    "D[h, y] = 0 | y = -2.5",
    "D[h, y] = 0 | y = 2.5",
    "h = 2.0 * (Sqrt[x ** 2 + y ** 2] < radius) + 1.0 * (Sqrt[x ** 2 + y ** 2] > radius)  | t = 0",
    "u = 0 | t = 0",
    "v = 0 | t = 0"
]

net = fnn.FNN(layers, activation, in_d=3, out_d=3, initializer=initializer)
start_time = time.time()
train(input=input, output=output, target={"h": h_train}, evaluation={"x": x_eval, "y": y_eval, "t": t_eval, "h": h_eval}, constant=constant, icbc=icbc, pde=pde, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=1000)
print(time.time() - start_time)