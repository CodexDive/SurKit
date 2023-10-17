.. Easy Surrogate documentation master file, created by
   sphinx-quickstart on Mon Mar 20 13:47:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SurKit's documentation!
==========================================
A surrogate model is a simplified model that can replace a complex or high-fidelity model in engineering problems.
Surrogate models can reduce the computational cost and time of evaluating the original model,
and facilitate the optimization and approximation processes.
SurKit allows users who are unfamiliar with neural networks to get started quickly by selecting the type of surrogate model
and providing parameters to automatically generate and train surrogate models.

User Guide
----------

.. toctree::
  :maxdepth: 2

  user/installation


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   example/solve_function
   example/uncertainty_quantification

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

..
   modules/surkit
.. toctree::
   :maxdepth: 2
   :caption: API

   modules/surkit.backend
   modules/surkit.data
   modules/surkit.conditions
   modules/surkit.pde
   modules/surkit.nn.pytorch
   modules/surkit.nn.oneflow
   modules/surkit.nn.jax
   modules/surkit.losses
   modules/surkit.optimizer
   modules/surkit.train

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

