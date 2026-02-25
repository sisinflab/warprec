.. _pipeline-train:

##################
Training Pipeline
##################

The Training Pipeline is the main experimental pipeline for full-scale experiments. It leverages **Ray Tune** for distributed hyperparameter optimization, supports cross-validation, computes statistical significance tests between models, serializes trained model checkpoints, and produces comprehensive result reports.

**Source:** ``warprec/pipelines/train.py`` — ``train_pipeline(path)``

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

When to Use
===========

- Running full benchmark experiments with HPO
- Comparing multiple models with statistical significance testing
- Producing reproducible experimental results with persistent artifacts
- Scaling training to multi-GPU or multi-node clusters

Prerequisites
=============

A **Ray cluster** must be running before invoking the Training Pipeline:

.. code-block:: bash

    ray start --head --num-cpus=16 --num-gpus=2

For granular per-trial resource control (RAM/VRAM limits):

.. code-block:: bash

    ray start --head --num-cpus=16 --num-gpus=2 \
        --resources='{"ram_gb": 64, "vram_gb": 48}'

For multi-node clusters, connect worker nodes with:

.. code-block:: bash

    ray start --address=<HEAD_IP>:6379

Configuration
=============

The Training Pipeline requires all configuration sections: ``reader``, ``writer``, ``splitter``, ``models``, and ``evaluation``.

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        data_type: transaction
        reading_method: local
        local_path: data/movielens.tsv
        rating_type: implicit

    writer:
        dataset_name: MyBenchmark
        writing_method: local
        local_experiment_path: experiments/benchmark/

    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
        validation_splitting:
            strategy: temporal_holdout
            ratio: 0.1

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

    dashboard:
        codecarbon:
            enabled: true
            save_to_file: true
            output_dir: ./carbon_reports/

Running
=======

.. code-block:: bash

    python -m warprec.run -c config/benchmark.yml -p train

Execution Flows
===============

The Training Pipeline supports three data split scenarios:

**1. Train/Test (no validation split)**

When only ``test_splitting`` is configured, HPO runs on the test split directly. Use this for simple experiments where a separate validation set is not needed.

**2. Train/Validation/Test**

When both ``test_splitting`` and ``validation_splitting`` are configured, HPO runs on the validation split, and the best model is evaluated on the held-out test set. This is the recommended setup.

**3. Cross-Validation**

When the splitting strategy is ``k_fold_cross_validation``, the pipeline trains and evaluates across all *k* folds. For each model, HPO runs on each validation fold, and the best hyperparameters are selected based on the average validation metric across folds.

HPO Strategies
==============

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Description
   * - ``grid``
     - Exhaustive grid search over all hyperparameter combinations.
   * - ``random``
     - Random sampling from the search space.
   * - ``hopt``
     - Bayesian optimization via HyperOpt (Tree-structured Parzen Estimators).
   * - ``optuna``
     - Bayesian optimization via Optuna with advanced pruning.
   * - ``bohb``
     - Bayesian Optimization and HyperBand for combined exploration and early stopping.

**Schedulers:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Scheduler
     - Description
   * - ``fifo``
     - First-In-First-Out. Runs all trials to completion.
   * - ``asha``
     - Asynchronous Successive Halving. Aggressively prunes underperforming trials based on intermediate results.

Search Space Syntax
-------------------

Hyperparameter search spaces are defined inline in the YAML configuration:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Syntax
     - Meaning
   * - ``[64, 128, 256]``
     - Discrete choice among values.
   * - ``[uniform, 0.001, 0.1]``
     - Continuous uniform sampling between min and max.
   * - ``[loguniform, 0.0001, 0.1]``
     - Log-uniform sampling (for learning rates).
   * - ``[randint, 1, 10]``
     - Random integer between min and max.
   * - ``[choice, adam, sgd, rmsprop]``
     - Categorical choice.

Output Artifacts
================

The Training Pipeline persists the following artifacts via the Writer module:

- **Results:** Metric scores for the best configuration of each model (CSV).
- **Per-user metrics:** Granular per-user scores (CSV) when ``per_user: true`` is configured.
- **Recommendations:** Top-K recommendation lists for each user (CSV).
- **Model checkpoints:** Serialized model weights (``.pth``) for the best configuration.
- **Hyperparameters:** Optimal hyperparameters per model (JSON).
- **Statistical significance:** Paired test results and correction tables (CSV).
- **Time reports:** Execution timing and CodeCarbon energy reports (when enabled).
