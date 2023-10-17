#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
os.environ['SRK_BACKEND'] = 'pytorch'
from matplotlib import pyplot as plt

import surkit.backend


import numpy as np

from surkit.nn import nn_ensemble
from surkit.train import train_gaussian
from surkit.data.sampling import uniform_sampler

layers = [32, 32]
activation = "Sigmoid"
optimizer = "Adam"
lr = 1e-4
x = uniform_sampler(-4, 4, 2000)
y = np.sin(x)
input = {"x": x}
target = {"y": y}
output = ["y"]
max_iteration = 10000
save_path = None
n_models = 8

net = nn_ensemble.MixtureNN(n_models, layers, activation, in_d=1, out_d=1)

train_gaussian(input=input, output=output, target=target, net=net, iterations=max_iteration,
               optimizer=optimizer, lr=lr, path=save_path, report_interval=1000)

x_show = uniform_sampler(-5, 5, 2000)
mean, var = net(surkit.backend.np_to_tensor(x_show))
mean = mean.cpu().detach().numpy()
var = var.cpu().detach().numpy()

plt.cla()
plt.plot(x_show, np.sin(x_show))
plt.plot(x_show, mean,c='red')
plt.fill_between(x_show.squeeze(), (mean-3*np.sqrt(var)).squeeze(), (mean+3*np.sqrt(var)).squeeze() ,color='grey',alpha=0.3)
plt.show()

