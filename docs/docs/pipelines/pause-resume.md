# Pause & Resume

A WarpRec experiment can run for hours or days: several models, each with a full hyperparameter sweep, followed by evaluation, recommendation generation and serialization. WarpRec lets you **stop such a run with a signal and resume it later**, continuing from where it stopped rather than from the beginning.

This is supported by the [training](training.md) and [swarm](swarm.md) pipelines, and configured through the [run](../configuration/run.md) section.

## What Gets Saved

Resume works at two layers, and both are needed.

**Trial layer — owned by Ray Tune.** Each model's sweep runs in a Ray Tune experiment with a deterministic name, `<run_name>__<model_name>`, under the storage path derived from the writer configuration. When a run is paused, Ray Tune checkpoints its experiment state, and every trial keeps the checkpoints written according to `checkpoint_to_keep`. On resume, unfinished trials continue from their last checkpoint instead of restarting.

**Pipeline layer — owned by WarpRec.** Ray Tune knows whether a sweep finished, but not whether the stages that follow it — evaluation, writing results, generating recommendations, serializing the model — also finished. WarpRec records that in a manifest:

```
<experiment_path>/run_state/<run_name>.json
```

The manifest tracks, for every model, one of five states: `pending`, `interrupted`, `hpo_completed`, `completed` or `failed`. On resume, `completed` models are skipped entirely, `hpo_completed` models skip the sweep and rebuild their best model from its checkpoint, and `interrupted` models continue their sweep.

The manifest also pins the timestamp embedded in the run's output file names, so a resumed run keeps merging into the same `Overall_Results_*.tsv`, `Overall_Params_*.json` and `Time_Report_*.tsv` as the run that created them, instead of starting a second set.

## Pausing a Run

Send `SIGINT` or `SIGTERM` to the WarpRec process:

```bash
# Interactively
Ctrl+C

# From another shell, or from a job scheduler before it reclaims the node
kill -TERM <pid>
```

WarpRec stops at the next safe point, saves the run state and exits with status **0** — a pause is not a failure. Sending a second signal aborts immediately without saving.

!!! warning
    Before this feature existed, interrupting a sweep produced a *plausible but incorrect* result: Ray Tune returns from an interrupted sweep normally rather than raising, so WarpRec selected a "best" model from the trials that happened to have reported by then and wrote it out as if the search had finished. WarpRec now detects the partial sweep and pauses instead of reporting a result it cannot stand behind.

## Resuming a Run

Run exactly the same command again:

```bash
warprec -c config.yml -p train
```

With `resume: auto` (the default), WarpRec finds the saved state, reports which models it is skipping and continues. Set `resume: force` to make a missing or incompatible state a hard error rather than a silent fresh start — see [Run Configuration](../configuration/run.md).

## Worked Example

```yaml
run:
    name: ml1m_baselines
    resume: auto
```

First run, interrupted with `Ctrl+C` while the second model is being optimized:

```
Run name: ml1m_baselines
...
Pause requested. WarpRec will stop at the next safe point and save the run state.
Hyperparameter optimization for MultiVAE was interrupted. No best model will be selected from this partial sweep.
Run 'ml1m_baselines' has been paused. Progress is saved at .../run_state/ml1m_baselines.json
Resume it by running the same command again with run.name set to 'ml1m_baselines' and run.resume set to 'auto' or 'force'.
```

Second run, resuming:

```
Run name: ml1m_baselines
Resuming run 'ml1m_baselines' created at 2026-09-02T10:00:00.
Skipping ItemKNN: already completed in this run.
Found a resumable Ray Tune experiment at .../ml1m_baselines__MultiVAE. Unfinished trials will continue from their last checkpoint.
```

## Cross-Validation

Under cross-validation, the sweep produces aggregated best hyperparameters rather than a single best checkpoint, and the final model comes from a separate retraining step. The manifest therefore records the best hyperparameters, and a resume that lands between the sweep and the retraining re-runs **only the retraining**, not the sweep.

## Swarm

The swarm pipeline supports pause and resume, with a **weaker guarantee** than the training pipeline.

In `swarm`, each model's tuning loop runs inside a Ray task on a worker process, so a signal delivered to the driver never reaches Ray Tune's own graceful handler. The driver cancels the running tasks instead and relies on Ray Tune's *periodic* experiment-state checkpoint. A swarm pause can therefore lose progress back to the last such checkpoint. Individual trial checkpoints are unaffected, so the loss is bounded and typically small.

For a run you expect to pause deliberately, prefer the training pipeline.

## Statistical Significance

Paired significance tests compare every model in the experiment and need each model's per-user metric tensors. On a resumed run, models skipped as `completed` are never re-evaluated, so those tensors would not be in memory.

When statistical significance is requested, WarpRec persists each model's evaluation results under `run_state/<run_name>/eval/` and reloads them on resume, so the tests still cover the whole experiment. If a file is missing or cannot be read, the model is excluded from the tests and a warning says so, rather than a quietly incomplete comparison being written.

Significance tests never run on a paused run — only on one that reaches the end.

## Cluster Preemption

On Kubernetes or a batch scheduler, arrange for the pod or job to receive `SIGTERM` before it is reclaimed, and give it enough grace time to reach a safe point. The run pauses cleanly and the next submission resumes it. Pair this with `resume: force` so that a misconfigured resume fails loudly instead of restarting the experiment from scratch.

## Upgrading

Three changes are visible to users of earlier WarpRec versions.

- **Ray Tune experiment directories are now named** `<run_name>__<model_name>` instead of randomly generated. Tooling that parsed those paths will see different names.
- **An interrupted sweep no longer produces a "best" model.** This is a bug fix, but it changes behaviour: a workflow that relied on interrupting a run and reading the results now gets a paused run and explicit resume instructions instead.
- **`Trainer`'s return types changed.** `Trainer.train_single_fold` and `Trainer.train_multiple_fold` return a `TrainingOutcome` object instead of a tuple, and `Trainer._load_best_model` is now the public `Trainer.load_best_model`. This affects only code that drives `Trainer` directly rather than through a pipeline.
