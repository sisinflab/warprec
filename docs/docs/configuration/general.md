# General Configuration

The **General Configuration** section defines parameters that affect the overall behavior of WarpRec.
It allows customization of numerical precision, device usage, logging verbosity, and integration of custom models or callbacks.

## Available Keywords

- **device**: Device used for training and evaluation. Supports `cpu` or `cuda` devices. Defaults to `cpu`.
- **backend**: Backend to use for data reading and writing. Supports `polars` or `pandas`. Defaults to `polars`.
- **ray_verbose**: Verbosity level of Ray Tune. Acceptable values are integers from 0 (silent) to 3 (very verbose). Defaults to `1`.
- **ray_address**: The address of the Ray cluster to connect to. Defaults to `auto`, which means that Warprec will automatically detect the Ray cluster.
- **cpu_data_prep**: The cpu resource to request for data preparation phase. Defaults to `1`.
- **custom_resources_data_prep**: The custom resources to request for data preparation phase.
- **label_selector_data_prep**: The label selector used for data preparation phase.
- **time_report**: Whether to write the `Time_Report` file, which tells apart the setup, the train-only time, the validation and the test evaluation of every model (see [Time Report](../data-management/writer.md#time-report)). Defaults to `True`.
- **gpu_deterministic**: Whether every process that trains or evaluates uses deterministic GPU kernels only, so that a run on a GPU, a resumed trial included, is bitwise reproducible. Accepts `True` or `False`. Defaults to `False`. It slows the computation down: see [GPU Determinism](#gpu-determinism).
- **custom_modules**: Python modules to import custom implementations into WarpRec. Can be a string or a list of strings. Each is imported under its file or directory name, so a name another module already has (a second `models.py`, or one of Python's own such as `json.py`) is refused rather than silently replaced by the other module.
- **callback**: Nested section to configure a custom callback.
- **azure**: Nested section to configure Azure information needed for reading and writing data from/to Azure Blob Storage.

## GPU Determinism

WarpRec seeds every model, so a run on the CPU is reproducible. A run on a GPU is not, seeds or no seeds: the CUDA kernels behind scatter and index operations (the gradients of embedding tables, `index_add_`, `scatter_add_`, the message passing of graph models) accumulate with atomic additions, in an order that changes from one run to the next, and floating-point addition is not associative. Two runs of the same trial on the same GPU therefore end on weights that differ in their last bits, and the difference can grow over the epochs.

With `gpu_deterministic: True`, every process that trains or evaluates — the Ray Tune trials, the cross-validation retraining, the test evaluation and the recommendation generation of the train and swarm pipelines, and the design, eval and estimate pipelines — before its first GPU operation:

- enables PyTorch's deterministic algorithms, with `torch.use_deterministic_algorithms(True, warn_only=True)`;
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which cuBLAS needs to be deterministic, unless the variable is already set (`:16:8` is accepted too, and uses less memory);
- keeps cuDNN deterministic and its benchmarking off, as WarpRec does for every model whether or not the flag is set.

**What it buys.** A trial that trains on a single GPU worker is bitwise reproducible: run twice, it ends on the same weights, and a trial paused and resumed ends on exactly the weights, best iteration and test results of the uninterrupted one (see [Pause & Resume](../pipelines/pause-resume.md)). The guarantee holds on the same GPU model with the same driver, CUDA and PyTorch versions. A trial spread over several GPU workers restores every worker's random state as well, but whether its gradient all-reduce is bitwise repeatable depends on the communication library (NCCL) and is not guaranteed by this flag.

**What it costs.**

- **Speed.** The deterministic implementations of scatter and index operations are slower than the atomic ones, sometimes much slower: models dominated by them, such as graph-based models and models with large embedding tables, train and evaluate noticeably slower.
- **Memory.** The cuBLAS workspace setting reserves a few tens of MiB of extra GPU memory in every process.
- **Operations without a deterministic implementation.** A few PyTorch operations have no deterministic CUDA kernel. WarpRec enables determinism in warn-only mode: such an operation still runs, with its usual kernel, and PyTorch emits a `UserWarning` naming it, instead of raising and failing the trial. A run that shows such a warning is not bitwise reproducible, and the warning says which operation is to blame.

**Without it** (the default), GPU runs match only up to floating-point nondeterminism: a resumed GPU trial ends on weights close to, but not bitwise equal to, those of the uninterrupted trial, exactly as two uninterrupted runs of it would. The metrics agree up to small numeric differences, which can occasionally change which trial a search selects as its best.

When the flag is `False`, WarpRec leaves PyTorch's determinism settings as they are, so a custom module that enables them itself keeps them. On the CPU, where runs are already reproducible, there is nothing to gain from it. Changing it does not invalidate a resume (see [What Invalidates a Resume](run.md#what-invalidates-a-resume)), but the epochs a trial trained before it was turned on stay as nondeterministic as they were: a resumed trial is bitwise equal to the uninterrupted one only when the flag was on from the start.

## Callback Configuration

The **callback** section allows pointing to a custom callback implementation and passing initialization parameters directly from the configuration:

- **callback_path**: Path to the Python script containing the callback implementation.
- **callback_name**: Name of the callback class. Must inherit from `WarpRecCallback`.
- **args**: List of positional arguments to pass to the callback constructor.
- **kwargs**: Dictionary of keyword arguments to pass to the callback constructor.

!!! important
    Custom callbacks must inherit from `WarpRecCallback` and be compatible with the provided arguments and keyword arguments. Follow this guide on how to implement your first callback.

## Azure Configuration

The **azure** section is required when using Azure Blob Storage for reading or writing data. The *remote-io* extra must be installed to enable this functionality. The section includes:

- **storage_account_name**: Name of the Azure Storage Account.
- **container_name**: Name of the Azure Blob container.

## Example General Configuration

Below is a complete example of a **general configuration** including precision, verbosity, and a custom callback:

```yaml
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
```
