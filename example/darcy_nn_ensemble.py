#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import datetime
import os
import time
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument('--backend', type=str, default="pytorch")
parser.add_argument('--cuda', type=str, default="3")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# choose backend from pytorch, oneflow, jax
os.environ['SRK_BACKEND'] = args.backend


import surkit.backend as bkd
import numpy as np
from surkit.nn import nn_ensemble
from surkit.train import train_gaussian

import h5py
data_path = "/mnt/nas_self-define/fukejie/dataset/2D_DarcyFlow_beta0.1_Train.hdf5"

np.random.seed(2300)
with h5py.File(data_path, "r") as h5_file:
    x = np.array(h5_file["x-coordinate"])
    y = np.array(h5_file["y-coordinate"])

    index = np.random.choice(range(128*128*10000 - 1), 1500000)

    nu = np.array(h5_file['nu']).reshape(-1, 1)[index]
    data = np.array(h5_file['tensor']).reshape(-1, 1)[index]

    inp = np.array(np.meshgrid(x, y)).T.reshape(-1, 2).repeat(10000,0)[index]


    split = int(0.75 * len(nu))

    x_train = inp[:split, 0].reshape(-1, 1)
    y_train = inp[:split, 1].reshape(-1, 1)
    nu_train = nu[:split]
    u_train = data[:split]

    x_eval = inp[split:, 0].reshape(-1, 1)
    y_eval = inp[split:, 1].reshape(-1, 1)
    nu_eval = nu[split:]
    u_eval = data[split:]



layers = [40] * 6
activation = "Sigmoid"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-4
input = {"x": x_train, "y": y_train, "nu": nu_train}
output = ["u"]
target={"u": u_train}
max_iteration = 10000
save_path = 'model/darcy_with_full_data_ensemble_%s' % (bkd.backend_name)
n_models = 8

net = nn_ensemble.MixtureNN(n_models, layers, activation, in_d=3, out_d=1)
start_time = time.time()
train_gaussian(input=input, output=output, target=target, evaluation={"x": x_eval, "y": y_eval, "nu": nu_eval, "u": u_eval}, net=net, iterations=max_iteration,
               optimizer=optimizer, lr=lr, path=save_path, report_interval=1000)
print(time.time() - start_time)

