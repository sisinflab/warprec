# Run Configuration

The **Run Configuration** section gives an experiment a stable identity and controls whether it can be **paused and resumed**. It is honoured by the [training](../pipelines/training.md) and [swarm](../pipelines/swarm.md) pipelines.

A long experiment — several models, each with a full hyperparameter sweep — can be stopped with a signal and picked up later, continuing from the last Ray Tune checkpoint instead of starting over. See [Pause & Resume](../pipelines/pause-resume.md) for the full workflow.

This section is optional. Omitting it gives every run an automatically derived name and the `auto` resume policy.

## Available Keywords

- **name**: Identifier of the run. When omitted, it is derived as `<dataset_name>_<fingerprint>`, where the fingerprint is the first 8 hexadecimal characters of a hash of the configuration. The resolved name is logged at the start of every run. Must start with a letter or a digit and contain only letters, digits, `.`, `_` and `-`, because it becomes a directory name locally and a blob key on Azure. Defaults to `None`.
- **resume**: The resume policy. One of `auto`, `force` or `never`. Defaults to `auto`.
- **errored_trials**: How to treat Ray Tune trials that errored before the pause. One of `skip`, `resume` or `restart`. Defaults to `skip`.
- **pause_on_signal**: Whether WarpRec installs the handlers that turn `SIGINT` and `SIGTERM` into a graceful pause. Defaults to `True`.

## Resume Policies

- **auto**: Resume when compatible saved state exists for the run name, otherwise start a fresh run. This is the everyday setting.
- **force**: Resume, and raise an error when no compatible saved state exists.
- **never**: Always start from scratch, discarding the saved state for this run name after a warning. The new run still records its own state, so a later `auto` can resume it.

!!! important
    In a batch script or a scheduled job, prefer `force` over `auto`. With `auto`, a typo in `name` silently starts a twelve-hour experiment from zero instead of resuming; with `force`, it fails immediately and tells you which path it searched.

## Errored Trial Policies

These map onto Ray Tune's two mutually exclusive restore flags, exposed as a single keyword so that an invalid combination cannot be expressed:

- **skip**: Leave errored trials as they are. Only unfinished trials are resumed.
- **restart**: Rerun each errored trial from scratch. Use this when the errors came from a bug you have since fixed.
- **resume**: Continue each errored trial from its last checkpoint. Use this when the errors were transient, such as a worker that ran out of memory on a node you have since replaced.

## What Invalidates a Resume

WarpRec fingerprints the configuration so that a resumed run cannot silently mix results produced under different settings.

Changing any of the following invalidates the **whole run**, because it changes the data or the meaning of the results:

- the `reader` section
- the `filtering` section
- the `splitter` section
- the `evaluation` section
- the set of model names

Changing a **single model's** parameters or search space invalidates only that model. Every other model keeps its saved state, and the changed model is optimized from scratch.

The following do **not** invalidate a resume, so a paused run can be resumed on a cluster of a different size or shape:

- `cpu_per_trial`, `gpu_per_trial`, `custom_resources_per_trial`, `label_selector`
- `max_concurrent_trials`, `num_workers`, and the model `device`
- `general.ray_verbose` and `general.ray_address`
- the whole `dashboard` section
- the `run` section itself

With `resume: auto`, an invalidated run starts fresh after a warning naming the mismatch. With `resume: force`, it raises instead.

## Example Run Configuration

```yaml
run:
    name: ml1m_baselines
    resume: force
    errored_trials: restart
    pause_on_signal: true
```
