# Design Pipeline

The Design Pipeline is optimized for **rapid prototyping and architectural validation**. It executes locally without Ray, trains each model exactly once with the specified hyperparameters (no HPO), and does not invoke the Writer module. This makes it the fastest way to test model implementations, validate configurations, and debug data pipelines.

**Source:** `warprec/pipelines/design.py` — `design_pipeline(path)`

---

## When to Use

- Testing a new model implementation before running full HPO
- Validating that a configuration file parses correctly and the data pipeline works end-to-end
- Debugging data loading, filtering, splitting, or evaluation issues
- Quick comparisons on small datasets

## Constraints

- **No HPO:** Each model must have **single-value hyperparameters** — search spaces (lists, ranges) are not allowed.
- **No Ray:** Runs entirely in the local process; a Ray cluster is not required.
- **No Writer:** Results are printed to the console but not persisted to disk. Use the Training Pipeline for persistent results.
- **Single split:** Uses one train/test split only (no cross-validation).

## Configuration

The Design Pipeline requires three configuration sections: `reader`, `splitter`, and `evaluation`. The `writer` section is ignored.

```yaml
reader:
    loading_strategy: dataset
    data_type: transaction
    reading_method: local
    local_path: data/movielens.tsv
    rating_type: implicit

splitter:
    test_splitting:
        strategy: temporal_holdout
        ratio: 0.1

models:
    EASE:
        l2: 10
    ItemKNN:
        k: 100
        similarity: cosine

evaluation:
    top_k: [10, 20]
    metrics: [nDCG, Precision, Recall, HitRate]
```

## Running

```bash
python -m warprec.run -c config/design.yml -p design
```

## Execution Flow

The `design_pipeline` function executes the following steps:

1. **Load configuration** — Parses the YAML file into a `WarpRecConfiguration` object.
2. **Initialize data** — Reads the dataset via the Reader module, applies filtering strategies, and splits the data.
3. **Create Evaluator** — Instantiates the `Evaluator` with all configured metrics and cutoff values.
4. **For each model:**

    a. Instantiate the model from the registry with the specified hyperparameters.

    b. If the model is iterative (inherits `IterativeRecommender`), train it via `train_loop()` for the configured number of epochs.

    c. Evaluate on the test set using all configured metrics.

    d. Fire `on_training_complete` and `on_evaluation_complete` callback hooks.

5. **Print results** — Display metric scores for all models.

!!! tip

    Use the Design Pipeline to iterate on model code. Once satisfied, switch to the Training Pipeline for full HPO by replacing single-value hyperparameters with search spaces and adding a `writer` section.
