# Estimate Pipeline

The Estimate Pipeline is designed for **pre-flight resource and runtime profiling**. It runs locally without Ray, expands discrete model configurations, and reports approximate training and evaluation costs before you launch a full experiment.

For iterative models, WarpRec measures a bounded number of train and evaluation batches and extrapolates end-to-end timings. For non-iterative models such as `ItemKNN` or `EASE`, WarpRec falls back to analytical train-space estimates exposed by the model implementation.

---

## When to Use

- Estimating RAM and optional VRAM requirements before running a large benchmark
- Comparing the expected cost of multiple model families on the same dataset
- Validating whether a configuration is feasible on a given CPU or GPU budget
- Profiling evaluation-time overhead independently from full HPO

## Constraints

- **No Ray:** Runs entirely in the local process.
- **Writer required:** Results are persisted through the Writer module as an estimate report.
- **Grid-like setups only:** Model configurations must expand to discrete setups. Continuous search spaces such as `[uniform, ...]` are not supported.
- **Optimization strategy:** If `optimization.strategy` is specified, it must be `grid`.
- **Experimental status:** The pipeline is marked experimental in the current implementation.

## Configuration

The Estimate Pipeline requires the same core sections as the Evaluation Pipeline, plus an `estimate` block:

- `reader`
- `writer`
- `splitter`
- `models`
- `evaluation`
- `estimate`

Optionally, you can also provide:

- `filtering`
- `general`

```yaml
reader:
    loading_strategy: dataset
    data_type: transaction
    reading_method: local
    local_path: path/to/my/dataset.tsv
    rating_type: implicit
writer:
    dataset_name: MyEstimateRun
    writing_method: local
    local_experiment_path: experiments/
splitter:
    test_splitting:
        strategy: temporal_holdout
        ratio: 0.1
models:
    ItemKNN:
        k: [grid, 50, 100, 200]
        similarity: cosine
    BPR:
        optimization:
            strategy: grid
            device: cuda
        embedding_size: [grid, 64, 128]
        reg_weight: [grid, 0.0001, 0.001]
        batch_size: 4096
        epochs: 200
        learning_rate: [grid, 0.0005, 0.001]
evaluation:
    top_k: [10, 20]
    metrics: [nDCG, Precision, Recall]
    strategy: full
estimate:
    warmup_batches: 10
    train_batches: 100
    eval_batches: 100
```

### Estimate Settings

| Key | Default | Meaning |
|---|---|---|
| `warmup_batches` | `10` | Number of initial batches ignored before timing starts. |
| `train_batches` | `100` | Number of training batches measured for iterative models. |
| `eval_batches` | `100` | Number of evaluation batches measured during inference profiling. |

## Running

```bash
python -m warprec.run -c path/to/config.yml -p estimate
```

## Output Artifacts

The Estimate Pipeline writes an `Estimate_Report_<timestamp>` file under the experiment `evaluation/` directory. The report aggregates all successful setups for each model and includes:

- Measured train and evaluation batch counts
- Average and standard deviation of train batch times
- Estimated train epoch time and total training time
- Average and standard deviation of evaluation batch times
- Estimated evaluation time
- Min/avg/max/std RAM usage
- Min/avg/max/std VRAM usage when CUDA profiling is enabled
- Free-form notes indicating CPU-only runs or analytical estimates

## Notes on Interpretation

- Timing estimates are extrapolations from a bounded sample of batches rather than full runs.
- Non-iterative models do not execute training loops during estimation; their train-space values come from model-specific analytical formulas.
- If CUDA is not configured, VRAM fields are left empty and the report is produced with CPU-only notes.
