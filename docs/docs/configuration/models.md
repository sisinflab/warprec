# Models Configuration

The **Models Configuration** module defines how each model in your experiment should be trained.
WarpRec allows flexible configuration of **training settings**, including hyperparameter search, scheduling, and resource management.

This section is divided into several nested sections to provide detailed control over model training:

- **meta**: Meta parameters affecting model initialization and checkpoint handling.
- **optimization**: Hyperparameter optimization settings using Ray Tune.
- **early_stopping**: Optional strategy to stop trials that reach a plateau.
- **parameters**: Model-specific parameters.

## Meta Parameters

The **meta** section allows controlling aspects of the model that do not directly interfere with training:

- **save_model**: Whether to save the model in the experiment directory. Defaults to `False`.
- **save_recs**: Whether to save generated recommendations. Defaults to `False`.
- **load_from**: Path to pre-trained model weights to load. Defaults to `None`.

## Optimization Configuration

The **optimization** section defines how hyperparameter optimization is performed:

- **strategy**: Optimization strategy. Defaults to `grid`. Supported strategies:
    - `grid`: Exhaustive search across the entire search space.
    - `random`: Random search within the search space.
    - `hopt`: HyperOpt algorithm for efficient exploration.
    - `optuna`: Optuna algorithm for efficient exploration.
    - `bohb`: BOHB algorithm for efficient exploration.

- **scheduler**: Scheduling algorithm for trials. Defaults to `fifo`. Supported schedulers:
    - `fifo`: First In First Out.
    - `asha`: ASHA scheduler for optimized early stopping and trial pruning.
    - `bohb`: HyperBand scheduler for BOHB. Must be paired with the `bohb` strategy.
    - `median`: Median stopping rule. Stops a trial whose performance falls below the median of the trials seen so far.

- **lr_scheduler**: Scheduling algorithm to adjust the learning rate at run time. Defaults to `None`.
- **optimizer**: Optimizer to use during the training process. Defaults to `None`.
- **properties**: Nested section for strategy and scheduler parameters.
- **device**: Training device, e.g., `cpu` or `cuda`. Overrides global device.
- **cpu_per_trial**: Number of CPU cores allocated per trial. Defaults to `1`.
- **gpu_per_trial**: Number of GPUs allocated per trial. Defaults to `0`.
- **custom_resources_per_trial**: A dictionary containing custom resources to request per trial during optimization. Defaults to an empty dictionary.
- **max_concurrent_trials**: Maximum number of trials allowed to run concurrently. Defaults to `None`, in which case WarpRec estimates a safe cap from the current Ray cluster resources.
- **label_selector**: A dictionary containing a set of labels with respective rules.
- **num_workers**: Number of worker processes for data loading. Defaults to `None` (main process).
- **block_size**: Number of items to predict at once for efficiency. Defaults to `50`.
- **checkpoint_to_keep**: Number of checkpoints to retain in Ray. Defaults to `5`.

!!! Example "Advanced Resource Management & Node Affinity"
    WarpRec allows strict control over cluster scheduling through logical resources and label selectors.

    **1. Logical Resource Constraints (`custom_resources_per_trial`)**
    Use this to prevent Out-Of-Memory (OOM) errors by treating RAM as a consumable logical resource. First, provision the Ray node exposing its memory capacity:
    ```bash
    ray start --head --resources='{"ram_gb": 64}'
    ```
    Then, require a fraction of this resource in your configuration:
    ```yaml
    models:
        EASE:
            optimization:
                cpu_per_trial: 4
                custom_resources_per_trial:
                    ram_gb: 32
    ```
    *Result:* The Ray scheduler will strictly limit concurrent trials on this node to a maximum of 2, ensuring memory limits are respected.

    **2. Node Affinity via Label Selectors (`label_selector`)**
    Use this to enforce execution on specific hardware or environments (e.g., isolating development from production, or targeting specific GPU architectures). Provision a node with custom labels:
    ```bash
    ray start --node-ip-address=<node-ip> --labels='env=dev,storage=fast_ssd'
    ```
    Then, bind the model training to these labels:
    ```yaml
    models:
        EASE:
            optimization:
                cpu_per_trial: 4
                label_selector:
                    env: dev
    ```
    *Result:* The trial workers will only be scheduled on nodes explicitly labeled with `env=dev`.

    *Tip:* Ray automatically injects hardware labels. You can use `label_selector: {"ray.io/accelerator-type": "A100"}` to target specific GPU architectures without manual node labeling. For more details, refer to the [Ray Scheduling Documentation](https://docs.ray.io/en/latest/ray-core/scheduling/labels.html).


### LR Scheduler Section

Within WarpRec standard pipelines, you can use a learning rate scheduler to increase your model performance. To do so, you can pass the following parameters under the lr_scheduler configuration block:

- **name**: Name of the scheduler (e.g., StepLR, ReduceLROnPlateau).
- **params**: A dictionary of parameters expected by the specific scheduler.

An example of this configuration could be something like this:

```yaml
models:
    MyModel:
        optimization:
            lr_scheduler:
                name: StepLR
                params:
                    step_size: 10
                    gamma: 0.2
```

For further details about the scheduling algorithms and their parameters, you can check the original [PyTorch Guide](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

### Optimizer Section

Within WarpRec standard pipelines, you customize the optimizer used during training to fit your need. To do so, you can pass the following parameters under the optimizer configuration block:

- **name**: Name of the optimizer (e.g., Adam, AdamW).
- **params**: A dictionary of parameters expected by the specific optimizer.

An example of this configuration could be something like this:

```yaml
models:
    MyModel:
        optimization:
            optimizer:
                name: SGD
                params:
                    momentum: 0.001
                    dampening: 0.0001
```

For further details about the optimizers and their parameters, you can check the original [PyTorch Guide](https://docs.pytorch.org/docs/stable/optim.html#algorithms).

!!! Tip
    You can combine the use of a learning rate scheduler and an optimizer to boost model performance. Choosing the right optimizer or scheduler is a critical factor in achieving optimal convergence, directly impacting the effectiveness and stability of the learning process.

### Properties Section

The **properties** subsection provides additional parameters to the optimization strategy or scheduler:

- **mode**: Whether to maximize or minimize the validation metric. Accepted values: `min` / `max`. Defaults to `max`.
- **desired_training_it**: Defines the number of iterations for final training after cross-validation. Strategies: `median`, `mean`, `min`, `max`. Defaults to `median`.
- **seed**: Random seed for reproducibility. Defaults to `42`.
- **time_attr**: Attribute used to measure time in the scheduler. Used by the `asha`, `bohb` and `median` schedulers, all of which fall back to `training_iteration` when it is not provided.
- **max_t**: Maximum time units per trial. Required by the `asha` and `bohb` schedulers.
- **grace_period**: Minimum time units per trial. Required by the `asha` and `median` schedulers, and ignored by `bohb`, which has no grace period.
- **reduction_factor**: Reduction rate of the `asha` and `bohb` schedulers. Required by both.
- **min_samples_required**: Minimum number of trials the `median` scheduler computes its median over. Must be greater than `0`. Defaults to `None`, in which case the Ray default of `3` is kept. Ignored by the other schedulers.
- **min_time_slice**: How long a trial runs before yielding under the `median` scheduler, in the units of `time_attr`. Must be greater than or equal to `0`. Defaults to `None`, in which case the Ray default of `0` is kept. Ignored by the other schedulers.
- **hard_stop**: Whether the `median` scheduler stops trials outright. When `False`, trials are paused instead and resumed FIFO once the others have finished. Defaults to `None`, in which case the Ray default of `True` is kept. Ignored by the other schedulers.
- **n_startup_trials**: Number of random trials the Bayesian strategies (`hopt`, `optuna`) run before their surrogate model starts guiding the search. Must be greater than `0`. Defaults to `None`, in which case the default of the underlying library is kept. Ignored by the other strategies.

!!! Example "Warm-up for Bayesian Search Strategies"
    Bayesian strategies such as `hopt` and `optuna` build a surrogate model of the search space from the trials observed so far. Until that model has enough evidence to be reliable, they sample configurations at random: this initial phase is the **warm-up**. The `n_startup_trials` property defines how many trials belong to it.

    **Configuration Example:**
    ```yaml
    models:
        LightGCN:
            optimization:
                strategy: optuna
                num_samples: 50
                properties:
                    seed: 42
                    n_startup_trials: 15   # 15 random trials, then the sampler takes over

            # Model parameters
            embedding_size: [choice, 64, 128, 256]
            n_layers: [choice, 1, 2, 3]
            learning_rate: [loguniform, 1e-5, 1e-2]
    ```

    **How to choose the value:**

    1. A **short** warm-up lets the sampler start exploiting earlier, which helps when the budget defined by `num_samples` is small, but it risks locking the search around the first promising region.
    2. A **long** warm-up covers more of the space before committing, which pays off on wide or highly non-convex search spaces, at the cost of spending a larger share of the budget on random configurations.

    As a rule of thumb, keeping the warm-up between 10% and 30% of `num_samples` is a reasonable starting point.

    *Result:* You gain explicit control over the exploration/exploitation trade-off of the search, instead of relying on the default of the underlying library. Reproducibility is preserved in both cases, since `seed` is always propagated to the sampler.

!!! Warning
    `n_startup_trials` is only used by the `hopt` and `optuna` strategies. If you set it while using `grid`, `random` or `bohb`, WarpRec logs an attention message and ignores the value.

## Early Stopping

The **early_stopping** section optionally adds stopping criteria for each trial:

- **monitor**: Metric to monitor, e.g., `score` (validation metric) or `loss`.
- **patience**: Consecutive evaluations without improvement before stopping. Required if early stopping is enabled.
- **grace_period**: Minimum number of evaluations before early stopping can trigger.
- **min_delta**: Minimum change to consider as an improvement.

!!! Example "ASHA Scheduler for Efficient Trial Pruning"
    When running a large hyperparameter search (e.g., with `random`, `optuna`, or `hopt`), many trials will perform poorly from the start. Instead of waiting for all trials to finish, you can use the **ASHA (Asynchronous Successive Halving Algorithm)** scheduler to aggressively terminate bad trials early and allocate resources only to the most promising ones.

    To use ASHA, you must define the `scheduler` as `asha` and provide the required parameters inside the `properties` block: `max_t`, `grace_period`, and `reduction_factor`.

    **Configuration Example:**
    ```yaml
    models:
        LightGCN:
            optimization:
                strategy: optuna
                num_samples: 100
                scheduler: asha
                properties:
                    time_attr: training_iteration  # The metric used to track time/progress
                    max_t: 200                     # Maximum iterations a trial can run
                    grace_period: 20               # Minimum iterations before pruning begins
                    reduction_factor: 3.0          # Halving rate (keeps top 1/3 of trials)

            # Model parameters
            embedding_size: [choice, 64, 128, 256]
            n_layers: [choice, 1, 2, 3]
            learning_rate: [loguniform, 1e-5, 1e-2]
            epochs: 200
    ```

    **How this works in practice:**

    1. **`time_attr: training_iteration`**: Tells the scheduler to evaluate the progress of the trials based on the number of training epochs/iterations completed.
    2. **`grace_period: 20`**: Every single trial is guaranteed to run for at least 20 iterations. This prevents the scheduler from killing a model that just has a slow start.
    3. **`reduction_factor: 3.0`**: At iteration 20, the scheduler compares all running trials. Only the top 33% (1/3) of the trials are allowed to continue. The bottom 66% are permanently stopped. This process repeats at iterations 60 (20 * 3) and 180 (60 * 3).
    4. **`max_t: 200`**: The absolute maximum number of iterations any trial is allowed to reach. This should generally match your model's `epochs` parameter.

    *Result:* By pruning unpromising trials early, ASHA allows you to test 100 configurations in a fraction of the time and compute cost it would take using the standard `fifo` scheduler.

!!! Example "BOHB: Bayesian Optimization with HyperBand"
    !!! important
        BOHB requires extra dependencies that are not installed by default. Install them with `pip install "warprec[bohb]"` (or `poetry install --extras bohb`) before using either the `bohb` strategy or the `bohb` scheduler.

    **BOHB** combines two halves that only work together. The `bohb` *strategy* is the Bayesian half: it models the search space and proposes promising configurations. The `bohb` *scheduler* is the HyperBand half: it runs many configurations briefly, **pauses** the weak ones and gives their budget to the survivors.

    Unlike ASHA, which stops a bad trial permanently, BOHB pauses trials and can resume them later. Only the `bohb` strategy knows how to resume a paused trial and insert new configurations in its place, so the two must always be used together.

    **Configuration Example:**
    ```yaml
    models:
        LightGCN:
            optimization:
                strategy: bohb
                num_samples: 100
                scheduler: bohb
                properties:
                    time_attr: training_iteration  # The metric used to track time/progress
                    max_t: 200                     # Maximum iterations a trial can run
                    reduction_factor: 3.0          # Keeps the top 1/3 at each rung

            # Model parameters
            embedding_size: [choice, 64, 128, 256]
            n_layers: [choice, 1, 2, 3]
            learning_rate: [loguniform, 1e-5, 1e-2]
            epochs: 200
    ```

    **How this works in practice:**

    1. **`time_attr: training_iteration`**: Tells the scheduler to measure a trial's progress by the number of training iterations completed.
    2. **`reduction_factor: 3.0`**: At the end of each rung, only the top 33% (1/3) of trials continue to the next one. The rest are paused.
    3. **`max_t: 200`**: The largest budget any single trial may reach. This should generally match your model's `epochs` parameter.
    4. **The Bayesian half**: as trials report results, the `bohb` strategy builds a model of which configurations do well at each budget, and samples new trials from it rather than at random.

    !!! warning
        `grace_period` does not apply to `bohb` and is ignored if provided: the scheduler derives its own rung sizes from `max_t` and `reduction_factor`. Pairing `scheduler: bohb` with any other strategy raises a configuration error, because the paused trials would never be resumed.

    *Result:* BOHB reaches strong configurations faster than random search plus HyperBand alone, because the budget it saves by pausing weak trials is reinvested in configurations its Bayesian model considers promising.

!!! Example "Median Stopping Rule"
    The **median stopping rule** is the simplest of the pruning schedulers: at each point in time it compares a trial against the median of every trial observed so far, and stops the ones falling below it. Unlike ASHA it has no rungs and no reduction factor, and unlike BOHB it works with any search strategy.

    It suits searches where trials are comparable at the same point in training and you want a cheap, predictable rule rather than a budget-allocation scheme.

    **Configuration Example:**
    ```yaml
    models:
        LightGCN:
            optimization:
                strategy: optuna
                num_samples: 100
                scheduler: median
                properties:
                    time_attr: training_iteration  # The metric used to track time/progress
                    grace_period: 20               # Trials younger than this are never stopped
                    min_samples_required: 5        # Median is only computed once 5 trials reported
                    hard_stop: true                # Stop trials outright rather than pausing them

            # Model parameters
            embedding_size: [choice, 64, 128, 256]
            n_layers: [choice, 1, 2, 3]
            learning_rate: [loguniform, 1e-5, 1e-2]
            epochs: 200
    ```

    **How this works in practice:**

    1. **`grace_period: 20`**: A trial is never stopped before iteration 20, so a slow starter is not killed prematurely.
    2. **`min_samples_required: 5`**: Until five trials have reported, there is no meaningful median and nothing is stopped. Raising this makes the rule more conservative early in a search.
    3. **`hard_stop: true`**: Stopped trials end for good. Set it to `false` to pause them instead: paused trials are resumed FIFO once every other trial has finished, which is useful when a late bloomer is still worth revisiting.

    !!! warning
        `max_t` and `reduction_factor` do not apply to `median` and are ignored if provided. `grace_period` and `min_time_slice` are expressed in the units of `time_attr`, so with the default `training_iteration` they count iterations, not seconds.

    *Result:* A cheap, strategy-agnostic pruning rule. It is less aggressive than ASHA at reallocating budget, but it makes no assumptions about the search algorithm and has only one required parameter.

## Example Model Configuration

In this section, we provide examples illustrating how to define the appropriate configuration for your experiment.

### Basic Configuration

The simplest way to define the model configuration is by directly specifying the parameters. Grid search is the default optimization strategy, so for each parameter, you can provide a list of values to explore, and WarpRec will manage the process automatically.

The following example demonstrates a basic grid search using the EASE model:

```yaml
models:
    EASE:
        l2: [10, 20, 30, 40, 50, 100, 150, 200]
```

An in depth configuration might include a model with more parameters and early stopping:

```yaml
models:
    BPR:
        early_stopping:
            patience: 20
            grace_period: 10
        embedding_size: [64, 128, 256]
        reg_weight: [0., 0.001, 1e-6]
        batch_size: [512, 1024, 2048, 4096]
        epochs: 300
        learning_rate: [0.001, 1e-4, 1e-5]
```

!!! note
    - Each model requires a separate configuration.
    - Trials of the same model can run in parallel; multiple models are trained sequentially.
    - Model parameters depend on the specific algorithm; consult the [Recommenders Documentation](../recommenders/index.md).

### Advanced Configuration

For advanced users, WarpRec provides support for sophisticated hyperparameter tuning and search space exploration, enabling efficient hyperparameter optimization and distributed experimentation.

Let's start from a really simple model configuration:

```yaml
models:
    LightGCN:
        embedding_size: 64
        n_layers: 2
        reg_weight: 0.0001
        batch_size: 512
        epochs: 50
        learning_rate: 0.001
```

This executes a grid search over a single parameter combination, effectively training just one model. Next, we will extend this example to explore a more comprehensive grid search:

```yaml
models:
    LightGCN:
        early_stopping:
            patience: 20
            grace_period: 10
        embedding_size: [64, 128, 256]
        n_layers: [1, 2, 3]
        reg_weight: [0., 1e-6]
        batch_size: [512, 1024, 2048]
        epochs: 200
        learning_rate: [0.001, 1e-4, 1e-5]
```

This configuration produces a total of 3 x 3 x 2 x 3 x 3 = 162 trials. Depending on the dataset size and available resources, the exploration may require some time. To optimize performance, you can leverage WarpRec's parallelization capabilities by adding the following to the configuration:

```yaml
models:
    LightGCN:
        optimization:
            cpu_per_trial: 4
            gpu_per_trial: 0.25
        early_stopping:
            patience: 20
            grace_period: 10
        embedding_size: [64, 128, 256]
        n_layers: [1, 2, 3]
        reg_weight: [0., 1e-6]
        batch_size: [512, 1024, 2048]
        epochs: 200
        learning_rate: [0.001, 1e-4, 1e-5]
```

With this setup, you can train up to 4 models at a time (if only 1 GPU is available), though this change will require more computational resources.

### Search Space Configuration

Advanced search algorithms (HyperOpt, Optuna) allow fine-grained exploration of hyperparameters. WarpRec supports multiple search spaces:

- `uniform` / `quniform`: Uniform distribution and quantized uniform distribution.
- `loguniform` / `qloguniform`: Logarithmic uniform distribution and quantized logarithmic uniform distribution.
- `randn` / `qrandn`: Random normal and quantized random normal.
- `randint` / `qrandint`: Random integers and quantized random integers.
- `lograndint` / `qlograndint`: Logarithmic random integers.
- `choice`: Default for discrete options.
- `grid`: Default for exhaustive grid search.

**Structure of parameter sampling in WarpRec**

Each parameter is defined as a list where:

1. `search_space` (str) - Name of the search space (e.g. `'uniform'`, `'qrandint'`, `'loguniform'`).
2. `min` (float/int) - Minimum value of the sampling range.
3. `max` (float/int) - Maximum value of the sampling range.
4. `quantization` (optional, float/int) - Step size for quantized spaces (e.g. `'qrandint'`, `'qloguniform'`). Only used for quantized spaces.
5. `log_base` (optional, int) - Base of the logarithm for log-scaled spaces (e.g. `'loguniform'`, `'qloguniform'`). Only used for log spaces.

The following examples illustrate how to sample values from these search spaces:

```yaml
param_1: ['uniform', 0.0, 1.0]
param_2: ['qrandint', 10, 500, 5]
param_3: ['qloguniform', 0.0, 1.0, 0.005, 2]
```

Let's now use the sampling spaces to create a more complex HPO and have more control over the parameter space:

```yaml
models:
    LightGCN:
        optimization:
            cpu_per_trial: 4
            gpu_per_trial: 0.25
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
```

This configuration performs hyperparameter optimization over 100 potential parameter combinations for the LightGCN model, executing 4 trials in parallel and applying early stopping.
