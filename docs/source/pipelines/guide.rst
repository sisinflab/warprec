.. _pipeline-guide:

######################
Pipeline Design Guide
######################

This guide explains how to design custom pipelines using WarpRec's modular engines, including best practices and common pitfalls.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Architecture Overview
=====================

Each WarpRec pipeline is a Python function that orchestrates the framework's five decoupled engines:

1. **Reader** — Loads raw data from local or cloud storage.
2. **Data Engine** — Filters, splits, and transforms data into model-ready structures.
3. **Recommendation Engine** — Instantiates, trains, and optimizes models.
4. **Evaluation** — Computes metrics and statistical tests.
5. **Writer** — Persists results, checkpoints, and reports.

A custom pipeline can selectively use any combination of these engines. For example, you might create a pipeline that only performs data analysis (Reader + Data Engine) or one that trains a single model and exports it for production (Reader + Data Engine + Recommendation Engine + Writer).

Creating a Custom Pipeline
===========================

A pipeline function follows this signature:

.. code-block:: python

    def my_pipeline(path: str):
        """Custom pipeline.

        Args:
            path: Path to the YAML configuration file.
        """
        ...

**Step 1: Load configuration.**

.. code-block:: python

    from warprec.utils.config import WarpRecConfiguration

    config = WarpRecConfiguration.from_yaml(path)

**Step 2: Initialize data.**

.. code-block:: python

    from warprec.data import initialize_data

    datasets = initialize_data(config)
    main_dataset = datasets["main"]

**Step 3: Create evaluator.**

.. code-block:: python

    from warprec.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        metric_list=config.evaluation.metrics,
        k_values=config.evaluation.top_k,
        train_set=main_dataset.train_sparse,
    )

**Step 4: Instantiate and train models.**

.. code-block:: python

    from warprec.recommenders import model_registry

    model_class = model_registry.get(model_name)
    model = model_class(params, info)

    # For iterative models:
    from warprec.recommenders.loops import train_loop
    train_loop(model, main_dataset, config)

**Step 5: Evaluate.**

.. code-block:: python

    results = evaluator.evaluate(model, dataloader, strategy, dataset, device)

Best Practices
==============

**Deterministic Seeding.**
Always set random seeds at the beginning of your pipeline to ensure reproducibility:

.. code-block:: python

    model.set_seed(config.general.seed)

**Callback Integration.**
Fire callback hooks at appropriate lifecycle stages so that custom callbacks (logging, visualization, etc.) work with your pipeline:

.. code-block:: python

    callback.on_training_complete(model)
    callback.on_evaluation_complete(model, params, results)

**Checkpoint Strategy.**
Save model checkpoints after training to enable later evaluation via the Evaluation Pipeline:

.. code-block:: python

    state = model.get_state()
    torch.save(state, checkpoint_path)

**Resource Cleanup.**
When using Ray, ensure that the cluster resources are properly released after the pipeline completes.

Common Pitfalls
===============

- **Forgetting to mask training interactions during evaluation.** The Evaluator automatically masks training items from predictions when provided with ``train_set``. Do not skip this parameter.
- **Using search spaces in the Design Pipeline.** The Design Pipeline expects single-value hyperparameters. Lists or ranges will cause a configuration validation error.
- **Running the Training Pipeline without a Ray cluster.** The Training Pipeline requires an active Ray cluster. Use the Design Pipeline for local testing.
- **ProxyRecommender in distributed mode.** ``ProxyRecommender`` must execute locally; it will raise an error if distributed Ray resources are configured. Use it only in the Design or Evaluation pipelines.
- **Mismatched hyperparameters when loading checkpoints.** When using ``meta.load_from``, ensure the model hyperparameters in the config match those used during training. Mismatches will cause ``load_state_dict()`` errors.
