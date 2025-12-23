.. _general:

########################
General Configuration
########################

The **General Configuration** section defines parameters that affect the overall behavior of WarpRec.
It allows customization of numerical precision, device usage, logging verbosity, and integration of custom models or callbacks.

-----------------------------
Available Keywords
-----------------------------

- **precision**: Numerical precision for computations during the experiment. Defaults to ``float32``. Higher precision (e.g., ``float64``) increases memory usage.
- **device**: Device used for training and evaluation. Supports ``cpu`` or ``cuda`` devices. Defaults to ``cpu``.
- **ray_verbose**: Verbosity level of Ray Tune. Acceptable values are integers from 0 (silent) to 3 (very verbose). Defaults to ``1``.
- **time_report**: Whether to report the time taken by each step. Defaults to ``True``.
- **train_data_preparation**: Which mode to use to prepare data for the experiment. This field supports the following values:
    - *None*: Leaving the field empty will result in the framework not pre-computing data structures needed for train. Each trial will compute the needed data.
    - *conservative*: In 'conservative' mode the framework will pre-compute all data structures needed only for the current model in exam, clearing the cache when passing to the next one.
    - *experiment*: In 'experiment' mode the framework will pre-compute all data structures needed for all the models in the configuration.

.. important::
    The **train_data_preparation** parameter affects how memory is managed during the experiment. The default mode uses the least memory, conservative requires more, and experiment is the most memory-intensive option. Choosing the latter modes can lead to faster training times, as key results are cached directly within the dataset.

- **custom_models**: Python modules to import custom models into WarpRec. Can be a string or a list of strings.
- **callback**: Nested section to configure a custom callback.
- **azure**: Nested section to configure Azure information needed for reading and writing data from/to Azure Blob Storage.

.. warning::
    Increasing the **precision** of computations (e.g., using ``float64`` instead of ``float32``) may significantly increase memory usage. For most experiments, ``float32`` is sufficient.

Callback Configuration
----------------------

The **callback** section allows pointing to a custom callback implementation and passing initialization parameters directly from the configuration:

- **callback_path**: Path to the Python script containing the callback implementation.
- **callback_name**: Name of the callback class. Must inherit from ``WarpRecCallback``.
- **args**: List of positional arguments to pass to the callback constructor.
- **kwargs**: Dictionary of keyword arguments to pass to the callback constructor.

.. important::
    Custom callbacks must inherit from ``WarpRecCallback`` and be compatible with the provided arguments and keyword arguments. Follow this guide on how to implement your first callback.

Azure Configuration
-------------------

The **azure** section is required when using Azure Blob Storage for reading or writing data. The *remote-io* extra must be installed to enable this functionality. The section includes:

- **storage_account_name**: Name of the Azure Storage Account.
- **container_name**: Name of the Azure Blob container.

Example General Configuration
-----------------------------

Below is a complete example of a **general configuration** including precision, verbosity, and a custom callback:

.. code-block:: yaml

   general:
       device: cuda
       ray_verbose: 0
       callback:
           callback_path: path/to/the/script.py
           callback_name: class_name
           args: [arg_1, arg_2]
           kwargs:
               kwargs_1: kwargs_value_1
               kwargs_2: kwargs_value_2
