# Quick Start

WarpRec provides a modular and extensible environment designed to support both advanced practitioners and newcomers.
This guide demonstrates two distinct workflows: local rapid prototyping and distributed training with hyperparameter optimization.

---

## Dataset Preparation

The first prerequisite is to structure the dataset in a **WarpRec-compatible input format**.
By default, WarpRec expects a **tab-separated values (.tsv)** file, where:

- the **first row** specifies the column headers,
- each **subsequent row** encodes a single user-item interaction event.

At minimum, the dataset must include the following fields:

- **user_id**: unique identifier of the user.
- **item_id**: unique identifier of the item.

In this minimal configuration, each row corresponds to a **binary implicit interaction**, all assigned equal weight.

Optionally, the dataset may also include:

- **rating**: a real-valued score representing the strength or relevance of the interaction.
  When present, WarpRec leverages this value as a **per-event weight** during training and evaluation.

- **timestamp**: a temporal marker (e.g., Unix epoch) specifying when the interaction occurred.
  This field is **mandatory** for temporal-based splitting strategies (e.g., holdout) and is also exploited by sequential or time-aware models.

| user_id | item_id | rating | timestamp |
|---|---|---|---|
| 1 | 1193 | 5 | 978300760 |
| 1 | 661 | 3 | 978302109 |
| 2 | 1357 | 5 | 978298709 |
| 3 | 2393 | 4 | 978297054 |

!!! note

    WarpRec provides multiple I/O backends through its **Reader** and **Writer** modules (e.g., local filesystem, Azure Blob Storage).
    In this guide, we demonstrate the simplest **local I/O workflow**.

    For a complete reference, see [Configuration](../configuration/index.md).

---

## Local Rapid Prototyping

This example uses the **Design Pipeline** for quick, local experimentation without hyperparameter optimization. It is ideal for testing model implementations, validating configurations, and debugging.

!!! important

    The Design Pipeline does **not** require a Ray cluster. It runs entirely locally.

**Step 1: Create the configuration file.**

Create a configuration file that fit your needs. An example could be something like this:

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
evaluation:
    top_k: [10, 20]
    metrics: [nDCG, Precision, Recall, HitRate]
```

!!! note

    In the Design Pipeline, each model must have **single-value hyperparameters** (no search spaces). This is because no HPO is performed: each model is trained exactly once with the specified configuration.

**Step 2: Run the experiment.**

```bash
python -m warprec.run -c path/to/config.yml -p design
```

WarpRec will:

1. Load and parse the dataset from the local path.
2. Split the data using a temporal holdout (90% train, 10% test).
3. Train both EASE and ItemKNN with the specified hyperparameters.
4. Evaluate both models and print the results.

This workflow is the fastest way to prototype and compare models.

---

## Distributed Training

This example uses the **Training Pipeline** with Ray-based distributed hyperparameter optimization. It demonstrates how WarpRec scales from a single machine to a multi-GPU cluster.

**Step 1: Start a Ray cluster.**

On your head node:

```bash
ray start --head --num-cpus=16 --num-gpus=2
```

For shared machines, restrict GPU visibility:

```bash
CUDA_VISIBLE_DEVICES=0,1 ray start --head --num-cpus=16 --num-gpus=2
```

**Step 2: Create the configuration file.**

When using a Ray cluster you can fully express WarpRec capabilities. In this case we propose a more configuration:

```yaml
reader:
    loading_strategy: dataset
    data_type: transaction
    reading_method: local
    local_path: path/to/my/dataset.tsv
    rating_type: implicit
writer:
    dataset_name: Benchmark
    writing_method: local
    local_experiment_path: experiments/
splitter:
    test_splitting:
        strategy: temporal_holdout
        ratio: 0.1
    validation_splitting:
        strategy: temporal_holdout
        ratio: 0.1
dashboard:
    wandb:
        enabled: true
        project: WarpRec-Benchmark
    codecarbon:
        enabled: true
        save_to_file: true
        output_dir: ./carbon_reports/
models:
    LightGCN:
        optimization:
            strategy: hopt
            scheduler: asha
            device: cuda
            cpu_per_trial: 4
            gpu_per_trial: 1
            num_samples: 20
        early_stopping:
            monitor: score
            patience: 10
            grace_period: 5
        embedding_size: [64, 128, 256]
        n_layers: [2, 3, 4]
        reg_weight: [uniform, 0.0001, 0.01]
        batch_size: 4096
        epochs: 200
        learning_rate: [uniform, 0.0001, 0.01]
    BPR:
        optimization:
            strategy: hopt
            scheduler: asha
            device: cuda
            cpu_per_trial: 4
            gpu_per_trial: 1
            num_samples: 20
        early_stopping:
            monitor: score
            patience: 10
            grace_period: 5
        embedding_size: [64, 128, 256, 512]
        reg_weight: [uniform, 0.0001, 0.01]
        batch_size: 4096
        epochs: 200
        learning_rate: [uniform, 0.0001, 0.01]
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
    validation_metric: nDCG@10
    strategy: full
    stat_significance:
        wilcoxon_test: true
        corrections:
            bonferroni: true
            fdr: true
            alpha: 0.05
```

**Key configuration highlights:**

- **Hyperparameter search spaces:** Lists (e.g., `[64, 128, 256]`) define discrete choices; `[uniform, min, max]` defines continuous ranges sampled by HyperOpt.
- **HyperOpt + ASHA:** Bayesian optimization with aggressive early stopping of underperforming trials.
- **Statistical testing:** Wilcoxon signed-rank test with Bonferroni and FDR corrections, automatically applied across all model pairs.
- **Green AI:** CodeCarbon tracks energy consumption and CO2 emissions per trial.

**Step 3: Run the experiment.**

```bash
python -m warprec.run -c path/to/config.yml -p train
```

Ray will distribute the HPO trials across available GPUs, pruning unpromising configurations early via ASHA. Results, carbon reports, and trained checkpoints are saved to the `experiments/industrial/` directory.

!!! tip

    For multi-node clusters, connect worker nodes to the head node with `ray start --address=<HEAD_IP>:6379`.
    See the [Cluster Management guide](../cloud/cluster-management.md) for setup details.

---

## Next Steps

- **Pipelines:** For a detailed guide to each execution pipeline (Design, Training, Evaluation), see [Pipelines](../pipelines/index.md).
- **Configuration Reference:** For a complete guide to all configuration keys, see [Configuration](../configuration/index.md).
- **Architecture:** For a deep dive into WarpRec's modular design, see [Architecture](architecture.md).
- **Models:** For the full catalog of 55 algorithms with hyperparameters, see [Recommenders](../recommenders/index.md).
- **Evaluation:** For the complete suite of 40 metrics, see [Evaluation](../evaluation/index.md).
