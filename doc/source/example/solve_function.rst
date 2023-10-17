======================
Solve Function
======================
We support either data-driven or physical constraints to solve the pde.
If we have enough data, we can use a data-driven strategy to train the neural network,
if we don't have data but know the form of the equations and the boundary conditions,
we can use a physically constrained strategy to train the neural network,
and if we have both the data and knowledge of the physics,
we can use a combination of the two to train the neural network

Chose Backend
=============
You can choose a backend from pytorch, oneflow and jax:

.. code-block:: python

    import os
    os.environ['SRK_BACKEND'] = 'pytorch'
    import surkit

or use command ``SRK_BACKEND='pytorch' python script.py`` to choose backend.

Define Formula
==============
If you choose the physical constraints strategy for training, then you need to provide the corresponding equations. For example: here we want to fit the 1D Burgers equation.

.. math::

    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}


    u(1, t) = u(-1, t) = 0


    u(x, 0) = -sin(\pi x)

You need to set up three lists: ``pde``, ``icbc`` and ``constant``, based on this formula, its initial & boundary conditions and constants that may exist that are replaced by designators. Please wrap the variables that appear in the formula with ``$``.

.. code-block:: python

    import numpy as np

    pde = ["D[u, t] + u * D[u, x] - nu * D[u, {x, 2}] = 0"]
    icbc = ["u = 0 | x = 1", "u = 0 | x = -1", "u = -Sin(Pi * x) | t = 0"]
    constant = {"nu": 0.01 / np.pi}
    output = ["u"]

Data Generation
===============
Next you need to prepare the training data, surkit provides 2 ways to get the data,
Sample data in user-defined domains,
or directly load user-generated datasets, but user need to inform which column of data corresponds to which variable in the formula.
Currently, we support ``csv`` files.

Sample from Domain
------------------
We support sampling a random set of data before each round of training:

.. code-block:: python

    x = {"low": -1, "high": 1, "size": (2000, 1)}
    t = {"low": 0, "high": 1, "size": (2000, 1)}
    input = {"x": x, "t": t}
    target = None

But if you only want to sample once, you can do it before training:

.. code-block:: python

    from surkit.data.sampling import random_sampler, uniform_sampler

    x = random_sampler(-1, 1, (2000, 1))
    t = random_sampler(0, 1, (2000, 1))
    input = {"x": x, "t": t}
    target = None

That gives us 2000 pairs of data.

Load Data from File
-------------------
If you choose the data-driven strategy for training, then you need to provide the dataset with the outputs.

.. code-block:: python

    header = ["x", "t", "u"]
    input = load_dataset("../dataset/data.csv", header)
    target = {}
    target[output] = input.pop(output)

Or you can process the data yourself by storing the input variable names and the input ``np.array`` in the ``input`` dictionary, and the output variable names and the output ``np.array`` in the ``target`` dictionary

.. code-block:: python

    input = {input_name1 : input_array1,  input_name2 : input_array2}
    target = {output_name1 : output_array1}

Define Model
============
Next we will choose a suitable neural network to train, here we use a fully connected neural network which requires to set the depth, width of hidden layers, activation function and parameter initializer.

.. code-block:: python

    from surkit.nn import fnn

    layers = [64, 64, 64, 64] # depth: 4, width: 64
    activation = "Tanh"
    initializer = "He normal"
    net = fnn.FNN(layers, activation, in_d=2, out_d=1, initializer=initializer)

Define Training
===============
Finally you need to set the training parameters and choose the appropriate ``train`` function based on the neural network type.

.. code-block:: python

    from surkit.train import train

    loss_function = "MSE"
    optimizer = "Adam"
    lr = 1e-4
    save_path = 'model/burgers.pth'
    max_iteration = 10000
    # If there is not data driven please set target to None, and similarly if there is
    # no formula constraint, please set pde, icbc, constant to None.
    train(input=input, output=output, target=target, constant=constant, icbc=icbc, pde=pde, net=net,
          iterations=max_iteration, optimizer=optimizer, lr=lr, loss_function=loss_function, path=save_path,
          report_interval=1000)

Load Model
==========
If you need to load a model that has already been trained, you need to use the ``backend.load`` function directly.
The information we save includes the network structure as well as the parameters.

.. code-block:: python

    import surkit.backend as bkd

    net = bkd.load('model/burgers.pth')

Inference
=========

The method of using the trained model is also simple, after preparing the data you can get the output directly using ``bkd.infer``.
You can get the following picture after visualising the output data.

.. code-block:: python

    from matplotlib import pyplot as plt, cm

    ms_t, ms_x = np.meshgrid(t, x)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)

    x_in = bkd.np_to_tensor(x).float()
    t_in = bkd.np_to_tensor(t).float()

    out = bkd.infer(net, bkd.cat([x_in, t_in],1))
    # visualisation
    u = out.data.cpu().numpy()
    pt_u0=u.reshape(256, 100)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.show()
    plt.close(fig)

.. image:: /example/burgers.png
    :alt: visualisation of burgers

Complete Code
=============

.. code-block:: python

    import os
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
        plt.show()
        plt.close(fig)


