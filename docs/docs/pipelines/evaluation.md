# Evaluation Pipeline

The Evaluation Pipeline is dedicated to **post-hoc analysis and inference** using pre-trained models, without retraining. It supports two evaluation modes: loading model checkpoints or reading precomputed recommendation files.

---

## When to Use

- Evaluating a saved checkpoint on new metrics or cutoff values
- Comparing models trained in different experiments
- Evaluating recommendations produced by external frameworks (e.g. Elliot, RecBole, ...)
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
    local_path: path/to/my/dataset.tsv
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
    ProxyRecommender:
        recommendation_file: results/external_recs.tsv
        separator: "\t"
        header: True
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
```

## Running

```bash
python -m warprec.run -c config/eval.yml -p eval
```

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
