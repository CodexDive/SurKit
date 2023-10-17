==========================
Uncertainty Quantification
==========================
For certain situations that require the use of uncertainty quantification such as data selection in active learning,
we also provide Neural Network Ensemble or Bayesian Neural Network to quantify the uncertainty.
In this example, we will attempt to fit a toy function ``y=sin(x)``.

.. code-block:: python

    import numpy as np
    from surkit.data.sampling import uniform_sampler
    x = uniform_sampler(-4, 4, 2000)
    y = np.sin(x)
    input = {"x": x}
    target = {"y": y}
    output = ["y"]

Neural Network Ensemble
=======================
Unlike other networks, neural network ensembles requires the additional parameter ``n_models``,
which represents how many sub-networks are included in this ensemble.
Also since only ``NLLloss`` is supported currently, there is no need to specify the loss function.
The output of the model consists of 2 parts, the mean of the sub-models' predictions and the variance of the sub-models' predictions.

.. code-block:: python

    from surkit.nn import nn_ensemble
    from surkit.train import train_gaussian

    layers = [32, 32]
    activation = "Sigmoid"
    optimizer = "Adam"
    lr = 1e-4
    max_iteration = 10000
    save_path = None
    n_models = 8

    net = nn_ensemble.MixtureNN(n_models, layers, activation, in_d=1, out_d=1)

    train_gaussian(input=input, output=output, target=target, net=net, iterations=max_iteration,
                   optimizer=optimizer, lr=lr, path=save_path, report_interval=1000)

.. image:: /example/ensemble.png
    :alt: visualisation of nn ensemble

Bayesian Neural Network
=======================
Bayesian neural networks also require three additional parameters ``noise_tol``, ``prior_mean``, ``prior_var``
representing noise tolerance and prior distribution of parameters respectively.
Training also requires an additional parameter ``samples``, which represents how many times the parameters need to be sampled for a prediction.
Also since only ``-ELBO`` is supported currently, there is no need to specify the loss function, and the learning rate needs to be increased appropriately.


.. code-block:: python

    from surkit.nn import bayes_nn
    from surkit.train import train_bayesian

    layers = [32, 32]
    activation = "Sigmoid"
    optimizer = "Adam"
    lr = 0.1
    max_iteration = 10000
    save_path = None
    samples = 8

    net = bayes_nn.BayesNN(layers, activation, in_d=1, out_d=1, noise_tol=.1, prior_mean=0., prior_var=1.)

    train_bayesian(input=input, output=output, target=target, net=net, iterations=max_iteration,
                   optimizer=optimizer, lr=lr, path=save_path, report_interval=1000, samples=samples)

.. image:: /example/bayesian_plot.png
    :alt: visualisation of bayesian nn

Complete Code
=============

NN-ensemble: `NN-ensemble <https://github.com/CodexDive/SurKit/example/exponential_ensemble.py>`_
BayesianNN: `BayesianNN <https://github.com/CodexDive/SurKit/example/bayesian.py>`_
