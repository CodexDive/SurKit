#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
os.environ['SRK_BACKEND'] = 'pytorch'
from matplotlib import pyplot as plt

import surkit.backend


import numpy as np

from surkit.nn import bayes_nn
from surkit.train import train_bayesian
from surkit.data.sampling import uniform_sampler

layers = [32, 32]
activation = "Sigmoid"
optimizer = "Adam"
lr = 0.1
x = uniform_sampler(-4, 4, 2000)
y = np.sin(x)
input = {"x": x}
target = {"y": y}
output = ["y"]
max_iteration = 10000
save_path = None
samples = 8

net = bayes_nn.BayesNN(layers, activation, in_d=1, out_d=1, noise_tol=.1, prior_mean=0., prior_var=1.)

train_bayesian(input=input, output=output, target=target, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, path=save_path, report_interval=1000, samples=samples)


x_show = uniform_sampler(-5, 5, 2000)
y_ = []
for i in range(samples):
    y_.append(net(surkit.backend.np_to_tensor(x_show)).cpu().detach().numpy())

y_train = np.array(y_)
mean = np.mean(y_train, 0)
var = np.var(y_train, 0)
plt.cla()
plt.plot(x_show, np.sin(x_show))
plt.plot(x_show, mean,c='red')
plt.fill_between(x_show.squeeze(), (mean-3*np.sqrt(var)).squeeze(), (mean+3*np.sqrt(var)).squeeze() ,color='grey',alpha=0.3)
plt.show()
