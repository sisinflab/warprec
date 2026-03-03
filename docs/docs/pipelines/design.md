# Design Pipeline

The Design Pipeline is optimized for **rapid prototyping and architectural validation**. It executes locally without Ray, trains each model exactly once with the specified hyperparameters (no HPO), and does not invoke the Writer module. This makes it the fastest way to test model implementations, validate configurations, and debug data pipelines.

---

## When to Use

- Testing a new model implementation before running full HPO
- Validating that a configuration file parses correctly and the data pipeline works end-to-end
- Debugging data loading, filtering, splitting, or evaluation issues
- Quick comparisons on small datasets

## Constraints

- **No HPO:** Each model must have **single-value hyperparameters** — search spaces (lists, ranges) **are not allowed**.
- **No Ray:** Runs entirely in the local process; a Ray cluster is not required.
- **No Writer:** Results are printed to the console but not persisted to disk. Use the Training Pipeline for persistent results.
- **Single split:** Uses one train/test split only (no cross-validation).

!!! tip

    Use the Design Pipeline to iterate on model code. Once satisfied, switch to the Training Pipeline for full HPO by replacing single-value hyperparameters with search spaces and adding a `writer` section.

## Configuration

The Design Pipeline requires three configuration sections: `reader`, `splitter`, and `evaluation`. The `writer` section is ignored.

```yaml
reader:
    loading_strategy: dataset
    data_type: transaction
    reading_method: local
    local_path: path/to/my/dataset.tsv
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
    MyCustomModel:
        arg_1: 10
        arg_2: 1e-3
        ...
evaluation:
    top_k: [10, 20]
    metrics: [nDCG, Precision, Recall, HitRate]
general:
    custom_modules: my_model.py
```

## Running

```bash
python -m warprec.run -c path/to/config.yml -p design
```
