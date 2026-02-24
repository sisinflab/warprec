#########################
Models Configuration
#########################

The **Models Configuration** module defines how each model in your experiment should be trained.
WarpRec allows flexible configuration of **training settings**, including hyperparameter search, scheduling, and resource management.

This section is divided into several nested sections to provide detailed control over model training:

- **meta**: Meta parameters affecting model initialization and checkpoint handling.
- **optimization**: Hyperparameter optimization settings using Ray Tune.
- **early_stopping**: Optional strategy to stop trials that reach a plateau.
- **parameters**: Model-specific parameters.

-----------------------------
Meta Parameters
-----------------------------

The **meta** section allows controlling aspects of the model that do not directly interfere with training:

- **save_model**: Whether to save the model in the experiment directory. Defaults to ``False``.
- **save_recs**: Whether to save generated recommendations. Defaults to ``False``.
- **load_from**: Path to pre-trained model weights to load. Defaults to ``None``.

-----------------------------
Optimization Configuration
-----------------------------

The **optimization** section defines how hyperparameter optimization is performed:

- **strategy**: Optimization strategy. Defaults to ``grid``. Supported strategies:
  - ``grid``: Exhaustive search across the entire search space.
  - ``random``: Random search within the search space.
  - ``hopt``: HyperOpt algorithm for efficient exploration.
  - ``optuna``: Optuna algorithm for efficient exploration.
  - ``bohb``: BOHB algorithm for efficient exploration.

- **scheduler**: Scheduling algorithm for trials. Defaults to ``fifo``. Supported schedulers:
  - ``fifo``: First In First Out.
  - ``asha``: ASHA scheduler for optimized early stopping and trial pruning.

- **lr_scheduler**: Scheduling algorithm to adjust the learning rate at run time. Defaults to ``None``.
- **properties**: Nested section for strategy and scheduler parameters.
- **device**: Training device, e.g., ``cpu`` or ``cuda``. Overrides global device.
- **cpu_per_trial**: Number of CPU cores allocated per trial. Defaults to ``1``.
- **gpu_per_trial**: Number of GPUs allocated per trial. Defaults to ``0``
- **num_workers**: Number of worker processes for data loading. Defaults to ``None`` (main process).
- **block_size**: Number of items to predict at once for efficiency. Defaults to ``50``.
- **checkpoint_to_keep**: Number of checkpoints to retain in Ray. Defaults to ``5``.

LR Scheduler Section
^^^^^^^^^^^^^^^^^^

Within WarpRec standard pipelines, you can use a learning rate scheduler to increase your model performance. To do so, you can pass the following parameters under the lr_scheduler configuration block:

- **name**: Name of the scheduler (e.g., StepLR, ReduceLROnPlateau).
- **params**: A dictionary of parameters expected by the specific scheduler.

An example of this configuration could be something like this:

.. code-block:: yaml

   models:
        MyModel:
            optimization:
                lr_scheduler:
                    name: StepLR
                    params:
                        step_size: 10
                        gamma: 0.2

For further details about the scheduling algorithms and their parameters, you can check the original `PyTorch Guide <https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_

Properties Section
^^^^^^^^^^^^^^^^^^

The **properties** subsection provides additional parameters to the optimization strategy or scheduler:

- **mode**: Whether to maximize or minimize the validation metric. Accepted values: ``min`` / ``max``. Defaults to ``max``.
- **desired_training_it**: Defines the number of iterations for final training after cross-validation. Strategies: ``median``, ``mean``, ``min``, ``max``. Defaults to ``median``.
- **seed**: Random seed for reproducibility. Defaults to ``42``.
- **time_attr**: Attribute used to measure time in the scheduler.
- **max_t**: Maximum time units per trial.
- **grace_period**: Minimum time units per trial.
- **reduction_factor**: ASHA scheduler reduction rate.

--------------
Early Stopping
--------------

The **early_stopping** section optionally adds stopping criteria for each trial:

- **monitor**: Metric to monitor, e.g., ``score`` (validation metric) or ``loss``.
- **patience**: Consecutive evaluations without improvement before stopping. Required if early stopping is enabled.
- **grace_period**: Minimum number of evaluations before early stopping can trigger.
- **min_delta**: Minimum change to consider as an improvement.

----------------------------
Example Model Configuration
----------------------------

In this section, we provide examples illustrating how to define the appropriate configuration for your experiment.

Basic Configuration
^^^^^^^^^^^^^^^^^^^

The simplest way to define the model configuration is by directly specifying the parameters. Grid search is the default optimization strategy, so for each parameter, you can provide a list of values to explore, and WarpRec will manage the process automatically.

The following example demonstrates a basic grid search using the EASE model:

.. code-block:: yaml

   models:
       EASE:
           l2: [10, 20, 30, 40, 50, 100, 150, 200]

A in depth configuration might include a model with more parameters and early stopping:

.. code-block:: yaml

   models:
       BPR:
           early_stopping:
               patience: 20
               grace_period: 10
           embedding_size: [64, 128, 256]
            weight_decay: [0., 0.001, 1e-6]
            batch_size: [512, 1024, 2048, 4096]
            epochs: 300
            learning_rate: [0.001, 1e-4, 1e-5]

.. note::
    - Each model requires a separate configuration.
    - Trials of the same model can run in parallel; multiple models are trained sequentially.
    - Model parameters depend on the specific algorithm; consult the :ref:`Recommenders Documentation <recommender>`.

Advanced Configuration
^^^^^^^^^^^^^^^^^^^^^^

For advanced users, WarpRec provides support for sophisticated hyperparameter tuning and search space exploration, enabling efficient hyperparameter optimization and distributed experimentation.

Let's start from a really simple model configuration:

.. code-block:: yaml

   models:
       LightGCN:
            embedding_size: 64
            n_layers: 2
            weight_decay: 0.0001
            batch_size: 512
            epochs: 50
            learning_rate: 0.001

This executes a grid search over a single parameter combination, effectively training just one model. Next, we will extend this example to explore a more comprehensive grid search:

.. code-block:: yaml

   models:
        LightGCN:
            early_stopping:
               patience: 20
               grace_period: 10
            embedding_size: [64, 128, 256]
            n_layers: [1, 2, 3]
            weight_decay: [0., 1e-6]
            batch_size: [512, 1024, 2048]
            epochs: 200
            learning_rate: [0.001, 1e-4, 1e-5]

This configuration produces a total of 3 x 4 x 2 x 3 x 3 = 216 trials. Depending on the dataset size and available resources, the exploration may require some time. To optimize performance, you can leverage WarpRec's parallelization capabilities by adding the following to the configuration:

.. code-block:: yaml

   models:
       LightGCN:
            optimization:
                parallel_trials: 5
            early_stopping:
               patience: 20
               grace_period: 10
            embedding_size: [64, 128, 256]
            n_layers: [1, 2, 3]
            weight_decay: [0., 1e-6]
            batch_size: [512, 1024, 2048]
            epochs: 200
            learning_rate: [0.001, 1e-4, 1e-5]

With this setup, you can train up to 5 models at a time, though this change will require more computational resources.

Search Space Configuration
--------------------------

Advanced search algorithms (HyperOpt, Optuna) allow fine-grained exploration of hyperparameters. WarpRec supports multiple search spaces:

- ``uniform`` / ``quniform``: Uniform distribution and quantized uniform distribution.
- ``loguniform`` / ``qloguniform``: Logarithmic uniform distribution and quantized logarithmic uniform distribution.
- ``randn`` / ``qrandn``: Random normal and quantized random normal.
- ``randint`` / ``qrandint``: Random integers and quantized random integers.
- ``lograndint`` / ``qlograndint``: Logarithmic random integers.
- ``choice``: Default for discrete options.
- ``grid``: Default for exhaustive grid search.

**Structure of parameter sampling in WarpRec**

Each parameter is defined as a list where:

1. ``search_space`` (str) - Name of the search space (e.g. ``'uniform'``, ``'qrandint'``, ``'loguniform'``).
2. ``min`` (float/int) - Minimum value of the sampling range.
3. ``max`` (float/int) - Maximum value of the sampling range.
4. ``quantization`` (optional, float/int) - Step size for quantized spaces (e.g. ``'qrandint'``, ``'qloguniform'``). Only used for quantized spaces.
5. ``log_base`` (optional, int) - Base of the logarithm for log-scaled spaces (e.g. ``'loguniform'``, ``'qloguniform'``). Only used for log spaces.

The following examples illustrate how to sample values from these search spaces:

.. code-block:: yaml

   param_1: ['uniform', 0.0, 1.0]
   param_2: ['qrandint', 10, 500, 5]
   param_3: ['qloguniform', 0.0, 1.0, 0.005, 2]

Let's now use the sampling spaces to create a more complex HPO and have more control over the parameter space:

.. code-block:: yaml

   models:
       LightGCN:
            optimization:
                parallel_trials: 5
                validation_metric: Recall@5
                strategy: hopt
                num_samples: 100
            early_stopping:
               patience: 20
               grace_period: 10
            embedding_size: [qrandint, 64, 320, 64]
            n_layers: [1, 2, 3]
            weight_decay: [uniform, 0.0, 1e-6]
            batch_size: [qrandint, 512, 10240, 512]
            epochs: 200
            learning_rate: [uniform, 1e-6, 1e-3]

This configuration performs hyperparameter optimization over 100 potential parameter combinations for the LightGCN model, executing up to 5 trials in parallel and applying early stopping.
