# Errors - API Reference

Every error WarpRec raises for something you can fix is a `WarpRecError`, defined in `warprec.errors`:

```python
from warprec import errors
from warprec.errors import ConfigurationError, WarpRecError
```

A failure inside WarpRec itself is not one of them: it stays a `RuntimeError` or an `AssertionError`, and a method a subclass must implement still raises `NotImplementedError`.

## The Hierarchy

```
WarpRecError(Exception)
├── ConfigurationError(WarpRecError, ValueError)
├── DataError(WarpRecError, ValueError)
│   └── DataNotFoundError(DataError, FileNotFoundError)
├── RegistryError(WarpRecError, ValueError)
├── ExtensionError(WarpRecError, ValueError)
│   └── CustomModuleError(ExtensionError, ImportError)
├── ModelError(WarpRecError, ValueError)
│   └── CheckpointError(ModelError)
├── EvaluationError(WarpRecError, ValueError)
├── RunStateError(WarpRecError, ValueError)
├── ClusterError(WarpRecError, ConnectionError)
└── ServingError(WarpRecError)
```

| Class | Raised when | Fields |
|---|---|---|
| `ConfigurationError` | The configuration is invalid or contradicts itself: a configuration file, model hyperparameters, search spaces, optimizer and scheduler names, filter and splitter parameters, an unknown pipeline name given to `warprec.run`. | `path`, `errors` |
| `DataError` | The data cannot support what was asked: a missing column, a dataset left empty by filtering, a user with too few unseen items to draw the sampled negatives from, side information that does not match the dataset. | |
| `DataNotFoundError` | A file WarpRec was asked to read does not exist: a dataset, a side-information file, a split, a configuration file, a checkpoint. | |
| `RegistryError` | A name is not registered (a model, metric, filter, optimizer, scheduler, search algorithm, re-ranker, similarity...), or a class is registered under a name another class already holds. When a registered name is close, the message ends with "Did you mean 'X'?". | `registry`, `name`, `available` |
| `ExtensionError` | Your code breaks a contract WarpRec relies on: a stash entry named like an argument WarpRec passes itself, a custom metric reporting a result of the wrong type or reading a block that is missing, a custom callback that is not a `WarpRecCallback`. | |
| `CustomModuleError` | A module listed under `general.custom_modules` cannot be imported, or its name is taken by another module. | |
| `ModelError` | A model cannot be built or used as asked: a model that needs the interactions, side information, a knowledge graph or multimodal features it was not given, a space estimate without the interactions it is computed from. | |
| `CheckpointError` | A checkpoint holds another model, was trained on other user or item ids than the dataset (`meta.load_from`), or lacks what serving it needs. | `path` |
| `EvaluationError` | A metric or an evaluation strategy cannot run on what it is given: an unknown strategy or propensity estimator, a metric that needs the training data or propensities it was not given, a multi-objective metric asked to combine a metric that reports no single score. | |
| `RunStateError` | Resuming a run conflicts with the state it saved: `run.resume: force` with no resumable state, or a state written by a newer WarpRec. | |
| `ClusterError` | The Ray cluster `general.ray_address` points at cannot be reached. | |
| `ServingError` | A request a served model cannot answer, with the HTTP status that says why. | `status`, `detail` |

A missing checkpoint file raises `DataNotFoundError`, not `CheckpointError`, so it is still a `FileNotFoundError` as it was before these classes existed.

An error raised while a configuration file is validated, such as the `RegistryError` of an unknown model name, is reported inside the one `ConfigurationError` loading the file raises (see below); the pydantic error it was collected in is that error's `__cause__`.

## Compatibility

Each class also derives from the built-in exception WarpRec raised before it existed, and the messages are unchanged, so code written against an earlier release keeps working:

```python
try:
    load_train_configuration("train.yml")
except ValueError:           # still catches a ConfigurationError
    ...
except FileNotFoundError:    # still catches a DataNotFoundError
    ...
```

Two things change for code that caught pydantic's own error:

- **Loading a configuration file** (`load_train_configuration`, `load_design_configuration`, `load_eval_configuration`, `load_estimate_configuration`, `load_serving_configuration`) raises one `ConfigurationError` instead of pydantic's `ValidationError`. `except ValueError` still catches it; `except pydantic.ValidationError` no longer does. The `ValidationError` is kept as its `__cause__`.
- **Model parameters** checked by name (`params_registry.get`, `model_param_from_dict`, the hyperparameters of every trial) raise a `ConfigurationError` keyed under `models.<name>` in the same way.

Building a configuration object directly, e.g. `TrainConfiguration(**data)`, still raises pydantic's `ValidationError`; the error a validator raised is in its `errors()[i]["ctx"]["error"]`, as one of the classes above.

## Reading a Configuration Error

A configuration file is validated whole, and the invalid keys are reported together, each by its dotted path and the validator's own message:

```
ConfigurationError: configs/train.yml is not a valid train configuration:
  splitter.test_splitting: You have chosen temporal_holdout splitting but the 'ratio' field has not been filled.
```

The checks that look across sections, such as the models against the registry and their hyperparameters, run once every section is valid on its own, so they are reported on the next attempt:

```
ConfigurationError: configs/train.yml is not a valid train configuration:
  models.BPR.learning_rate: Values of learning_rate for BPR model must be > 0. Values received as input: [-0.1]
```

The same pairs are in `error.errors`, a list of `(key, message)`, and the file in `error.path`. An error that belongs to no single key, such as an unknown model name, is listed without one.

## The Command Line

`python -m warprec.run` (and the `warprec` and `warprec.run` commands) and `warprec.serve` print a `WarpRecError` as one line, `<ClassName>: <message>`, on stderr, without a traceback, and exit with status 1. Pass `--debug` to raise it instead, so the full traceback shows:

```bash
python -m warprec.run -c config.yml -p train --debug
warprec.serve -c serve.yml --debug
```

Any other exception, one that means something went wrong inside WarpRec or a library it uses, is not caught: its traceback always shows.

## Across Ray

Every error pickles to an equal one, its fields included, so an error raised inside a Ray task or a training trial reaches the driver as its own class:

```python
@ray.remote
def validate(params):
    return params_registry.get("BPR", **params)

try:
    ray.get(validate.remote({"learning_rate": -1}))
except ConfigurationError as error:
    print(error.errors)
```

## Not an Error

`RunPaused` (`warprec.utils.pause`), raised when a run stops to be resumed later, is not a `WarpRecError`: a paused run has not failed.

## Reference

::: warprec.errors
    options:
      show_root_heading: false
      members_order: source
