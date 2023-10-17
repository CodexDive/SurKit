#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
os.environ['SRK_BACKEND'] = 'pytorch'
import numpy as np
import surkit

from matplotlib import pyplot as plt, cm

from surkit.nn import fnn
from surkit.train import train
import surkit.backend as bkd

layers = [64, 64, 64, 64]
activation = "Tanh"
initializer = "He normal"
loss_function = "MSE"
optimizer = "Adam"
lr = 1e-4
x = {"low": -1, "high": 1, "size": (2000, 1)}
t = {"low": 0, "high": 1, "size": (2000, 1)}
input = {"x": x, "t": t}
output = ["u"]
max_iteration = 10000
save_path = 'model/burgers_%s' % os.environ['SRK_BACKEND']
pde = ["D[u, t] + u * D[u, x] - nu * D[u, {x, 2}] = 0"]
icbc = ["u = 0 | x = 1", "u = 0 | x = -1", "u = -Sin(Pi * x) | t = 0"]
constant = {"nu": 0.01 / np.pi}

net = fnn.FNN(layers, activation, in_d=2, out_d=1, initializer=initializer)

train(input=input, output=output, target=None, constant=constant, icbc=icbc, pde=pde, net=net, iterations=max_iteration,
    optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path, report_interval=1000)

if bkd.backend_name in ['pytorch', 'oneflow', '']:
    net = bkd.load(save_path).cpu()

    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)

    ms_t, ms_x = np.meshgrid(t, x)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)

    x_in = bkd.np_to_tensor(x).float().cpu()
    t_in = bkd.np_to_tensor(t).float().cpu()

    out = bkd.forward(net, bkd.cat([x_in, t_in], 1))
    u = out.data.cpu().numpy()
    pt_u0 = u.reshape(256, 100)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.show()
    plt.close(fig)

if bkd.backend_name == 'jax':
    import jax
    from flax.training import train_state

    init_key = jax.random.PRNGKey(0)
    params = net.init(init_key, jax.random.uniform(init_key, (1, len(input))))
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=surkit.optimizer.get(optimizer)(learning_rate=lr))
    state = bkd.load(state, save_path)

    x = np.linspace(-1,1,256)
    t = np.linspace(0,1,100)

    ms_t, ms_x = np.meshgrid(t, x)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)

    x_in = bkd.np_to_tensor(x)
    t_in = bkd.np_to_tensor(t)

    out = bkd.forward(state, bkd.cat([x_in, t_in],1))
    u = out
    pt_u0 = u.reshape(256,100)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.show()
    plt.close(fig)
