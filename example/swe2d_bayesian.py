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
from surkit.nn import bayes_nn
from surkit.train import train_bayesian

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
    index = np.random.choice(range(len(h) - 1), 1500)

    x_train = inp[index[:1000], 0].reshape(-1, 1)
    y_train = inp[index[:1000], 1].reshape(-1, 1)
    t_train = inp[index[:1000], 2].reshape(-1, 1)
    h_train = h[index[:1000]]

    x_eval = inp[index[1000:], 0].reshape(-1, 1)
    y_eval = inp[index[1000:], 1].reshape(-1, 1)
    t_eval = inp[index[1000:], 2].reshape(-1, 1)
    h_eval = h[index[1000:]]

layers = [32] * 2
activation = "Sigmoid"
optimizer = "Adam"
lr = 1e-3
input = {"x": x_train, "y": y_train, "t": t_train}
output = ["h", "u", "v"]
target={"h": h_train}
max_iteration = 10000
save_path = 'model/swe2d_with_full_data_bnn_%s' % bkd.backend_name
samples = 8

net = bayes_nn.BayesNN(layers, activation, in_d=3, out_d=1, noise_tol=.1, prior_mean=0., prior_var=1.)
start_time = time.time()
train_bayesian(input=input, output=output, target=target, evaluation={"x": x_eval, "y": y_eval, "t": t_eval, "h": h_eval}, net=net, iterations=max_iteration,
               optimizer=optimizer, lr=lr, path=save_path, report_interval=1000, samples=samples)
print(time.time() - start_time)

