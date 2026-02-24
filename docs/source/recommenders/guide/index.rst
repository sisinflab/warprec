####################
Implementation Guide
####################

.. toctree::
   :hidden:
   :maxdepth: 2

   classic
   iterative

WarpRec provides a flexible and modular framework that allows you to implement your own
recommendation models. This guide is designed to help you understand how to create
custom recommenders and integrate them seamlessly into the WarpRec ecosystem.

The guide will focus on two main types of models:

1. **Classical Models**
   These are traditional recommendation algorithms that often rely on matrix operations,
   item similarity, or other non-iterative methods. They are typically fast to train
   and serve as a good starting point for exploring custom recommenders.

2. **Iterative Models**
   These models learn their parameters through iterative optimization, often involving
   multiple passes over the data and convergence criteria. Iterative models are more
   flexible and can capture complex patterns, but they may require more careful
   configuration and computational resources.

By exploring both classical and iterative approaches, you will gain a comprehensive
understanding of how to implement custom recommenders in WarpRec, and how to leverage
the framework's utilities for configuration management, device handling, and
integration with advanced workflows such as hyperparameter optimization.

General Usage
-------------

Besides the specific implementation details for each type of model, there are some general information that is useful to know when implementing a custom recommender.

Registering Your Model
^^^^^^^^^^^^^^^^^^^^^^

WarpRec uses a model registry to manage available models. Ensure that your class is decorated with ``@model_registry.register(name="MyModel")``. This allows the framework to discover and use the model automatically.

.. code-block:: python

    from warprec.utils.registry import model_registry

    @model_registry.register(name="MyModel")
    class MyModel(Recommender):
      # Rest of the model implementation...

Add the Model to your Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case you are using configuration files to execute your experiment, you need an extra step to ensure that WarpRec recognizes your model. To use your custom model in a WarpRec pipeline, add it to your configuration file under the ``general`` section. You have two options:

- Create a script with only the model definition and pass it to the configuration.
- Create a fully fledged module (which can also include other models and metrics) and pass it to the configuration.

.. code-block:: yaml

    general:
        custom_modules: [my_script.py, my_module]

.. note::

    WarpRec supports both relative and absolute paths for custom model scripts and modules.
    Also, you can pass a list of scripts/modules if needed. Organize your models as you prefer.
