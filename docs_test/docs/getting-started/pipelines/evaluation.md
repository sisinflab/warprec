# Evaluation Pipeline

The Evaluation Pipeline is dedicated to **post-hoc analysis and inference** using pre-trained models, without retraining. It supports two evaluation modes: loading model checkpoints or reading precomputed recommendation files.

**Source:** `warprec/pipelines/eval.py` — `eval_pipeline(path)`

---

## When to Use

- Evaluating a saved checkpoint on new metrics or cutoff values
- Comparing models trained in different experiments
- Evaluating recommendations produced by external frameworks (Elliot, RecBole, Cornac)
- Computing statistical significance between pre-trained models

## Prerequisites

- Pre-trained model checkpoints (`.pth` files) saved by the Training Pipeline, **or**
- Precomputed recommendation files (TSV/CSV) from WarpRec or other frameworks

A Ray cluster is **not** required for the Evaluation Pipeline.

## Evaluation Modes

### Mode 1: Checkpoint Loading

Load a pre-trained model checkpoint and evaluate it on the test set:

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
    LightGCN:
        meta:
            load_from: experiments/benchmark/LightGCN/checkpoint.pth
        embedding_size: 256
        n_layers: 3
        reg_weight: 0.001
        batch_size: 4096
        epochs: 200
        learning_rate: 0.001

evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate, MAP, MRR]
```

The `meta.load_from` key specifies the path to the saved checkpoint. The model is instantiated with the given hyperparameters, and then the saved weights are loaded via `torch.load()` and `model.load_state_dict()`.

### Mode 2: External Recommendation Files

Evaluate precomputed recommendation files using `ProxyRecommender`:

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
    ExternalModel:
        meta:
            model_name: ProxyRecommender
        recommendation_file: results/external_recs.tsv
        separator: "\t"
        header: true

evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
```

## Running

```bash
python -m warprec.run -c config/eval.yml -p eval
```

## Execution Flow

1. **Load configuration** — Parses the YAML file.
2. **Initialize data** — Reads and splits the dataset (to obtain the test set ground truth).
3. **Create Evaluator** — Instantiates with all configured metrics and cutoff values.
4. **For each model:**

    a. Instantiate the model from the registry.

    b. Load the checkpoint from `meta.load_from` (or load recommendation file for ProxyRecommender).

    c. Evaluate on the test set.

5. **Statistical significance** — If multiple models are evaluated and `stat_significance` is configured, compute pairwise statistical tests.
6. **Write results** — Persist results via the Writer module (if configured).

## Statistical Significance

The Evaluation Pipeline supports the same statistical significance testing as the Training Pipeline. When evaluating multiple models (either checkpoints or external recommendations), configure:

```yaml
evaluation:
    stat_significance:
        wilcoxon_test: true
        paired_t_test: true
        corrections:
            bonferroni: true
            fdr: true
            alpha: 0.05
```
