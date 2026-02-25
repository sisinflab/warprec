.. _architecture:

####################
Architecture
####################

WarpRec is designed around principles of **modularity** and **separation of concerns**. Five decoupled engines manage the end-to-end recommendation lifecycle, from data ingestion to model evaluation. An Application Layer exposes trained models through a REST API and an MCP agentic interface.

This architecture allows researchers to inject custom logic, swap data backends, or integrate individual modules into external pipelines, all without modifying the core framework.

.. image:: _static/warprec-architecture.png
   :alt: WarpRec Architecture
   :align: center
   :width: 100%

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

The Four Pillars
=================

WarpRec is built on four foundational principles that distinguish it from existing frameworks.

Scalability (Ray)
-----------------

WarpRec ensures strict code portability from local prototyping to industrial deployment. The framework supports execution from single-node to multi-GPU configurations, leveraging `Ray <https://docs.ray.io/>`_ for multi-node orchestration. This integration enables elastic scaling across cloud infrastructures, optimizing resource allocation and reducing computational costs. A model trained on a laptop can be deployed to a Ray cluster without changing a single line of configuration.

Green AI (CodeCarbon)
---------------------

WarpRec is the first Recommender Systems framework to enforce ecological accountability by natively integrating `CodeCarbon <https://mlco2.github.io/codecarbon/>`_. Every training trial can be profiled for real-time energy consumption (CPU, GPU, RAM) and CO2 emissions, enabling researchers to report the environmental cost of their experiments alongside traditional performance metrics.

Agentic Readiness (MCP)
------------------------

Anticipating the shift toward autonomous systems, WarpRec natively implements the `Model Context Protocol (MCP) <https://modelcontextprotocol.io/>`_ server interface. This transforms the recommender from a static predictor into a queryable tool that LLMs and autonomous agents can invoke dynamically within their reasoning loops, bridging the gap between recommendation and conversational AI.

Scientific Rigor (Statistical Testing)
---------------------------------------

WarpRec automates reproducibility and statistical validation. The evaluation suite includes paired tests (Student's t-test, Wilcoxon signed-rank) and independent-group analyses (Kruskal-Wallis, Mann-Whitney U), with automatic corrections for the Multiple Comparison Problem via Bonferroni, Holm-Bonferroni, and FDR (Benjamini-Hochberg) methods. All stochastic operations are anchored to global random seeds.

-----

Modular Engines
================

Reader Module
-------------

The Reader Module abstracts the complexity of data ingestion by leveraging `Narwhals <https://narwhals-dev.github.io/narwhals/>`_ as a backend-agnostic compatibility layer. Instead of enforcing a specific dataframe library, Narwhals allows the Reader to work seamlessly with **Pandas**, **Polars**, or **Spark** backends. This design avoids costly conversion overheads and lets users switch backends via a single configuration key:

.. code-block:: yaml

    general:
        backend: polars   # or pandas

The module supports ingestion from both local filesystems and cloud-based object storage (Azure Blob Storage), enabling researchers to transition from local prototyping to large-scale cloud experimentation without modifying their ingestion pipelines.

**Supported formats:** TSV, CSV, Parquet, JSON.

**Readers:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reader
     - Description
   * - ``LocalReader``
     - Reads data from the local filesystem.
   * - ``AzureBlobReader``
     - Reads data from Azure Blob Storage containers.

Data Engine
-----------

The Data Engine transforms raw transactions into refined, model-ready structures through three specialized components.

**Filtering.**
Includes 13 distinct strategies organized into three functional families:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Family
     - Strategy
     - Description
   * - Filter by Rating
     - ``MinRating``
     - Removes interactions below a minimum rating threshold.
   * -
     - ``UserAverage``
     - Removes interactions below the user's average rating.
   * -
     - ``ItemAverage``
     - Removes interactions below the item's average rating.
   * - K-Core / Interaction Count
     - ``UserMin``
     - Removes users with fewer than *n* interactions.
   * -
     - ``UserMax``
     - Removes users with more than *n* interactions.
   * -
     - ``ItemMin``
     - Removes items with fewer than *n* interactions.
   * -
     - ``ItemMax``
     - Removes items with more than *n* interactions.
   * -
     - ``IterativeKCore``
     - Iteratively removes users and items with fewer than *k* connections until convergence.
   * -
     - ``NRoundsKCore``
     - Applies k-core decomposition for a fixed number of rounds.
   * - Cold-Start Heuristics
     - ``UserHeadN``
     - Retains only the first *n* interactions per user (chronological).
   * -
     - ``UserTailN``
     - Retains only the last *n* interactions per user (chronological).
   * -
     - ``DropUser``
     - Removes specific users from the dataset.
   * -
     - ``DropItem``
     - Removes specific items from the dataset.

**Splitter.**
Partitions the filtered dataset using 6 distinct strategies to prevent data leakage and ensure rigorous evaluation:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Strategy
     - Description
   * - ``temporal_holdout``
     - Splits by time, placing the most recent interactions in the test set at a given ratio.
   * - ``temporal_leave_k_out``
     - Holds out the last *k* interactions per user based on timestamps.
   * - ``random_holdout``
     - Randomly splits interactions at a given ratio.
   * - ``random_leave_k_out``
     - Randomly holds out *k* interactions per user.
   * - ``timestamp_slicing``
     - Splits at a fixed timestamp boundary.
   * - ``k_fold_cross_validation``
     - Partitions data into *k* folds for cross-validated evaluation.

All stochastic partitioning is anchored to global random seeds, guaranteeing experimental reproducibility.

**Dataset.**
Orchestrates the transformation of raw partitions into high-performance internal representations. It unifies Interaction, Session, and Context Management to standardize sparse user-item signals, variable-length behavioral sequences, and auxiliary metadata. The Dataset enforces strict Data Alignment by mapping external identifiers to contiguous internal indices, ensuring referential integrity before materializing optimized training and evaluation structures.

Recommendation Engine
---------------------

This module governs the lifecycle and training of diverse model architectures. It includes two components: **Models** and **Trainer**.

**Models.**
WarpRec provides a repository of **55 built-in algorithms** spanning 6 fundamental classes:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Family
     - Count
     - Representative Models
   * - Unpersonalized
     - 3
     - Pop, Random, Proxy
   * - Content-Based
     - 1
     - VSM
   * - Collaborative Filtering
     - 29
     - EASE\ :sup:`R`, LightGCN, MultiVAE, BPR, NeuMF, ItemKNN, UserKNN, DGCF, MixRec, and more
   * - Context-Aware
     - 8
     - FM, DeepFM, Wide&Deep, xDeepFM, AFM, NFM, DCN, DCNv2
   * - Sequential
     - 10
     - SASRec, BERT4Rec, GRU4Rec, Caser, gSASRec, LinRec, and more
   * - Hybrid
     - 4
     - AddEASE, CEASE, AttributeItemKNN, AttributeUserKNN

For a complete list with hyperparameters and configuration examples, see the :ref:`Recommenders <recommender>` section.

**Trainer.**
The Trainer serves as the core execution engine, orchestrating model optimization and state persistence via automated checkpointing to enable seamless experiment resumption. Key capabilities:

- **Distributed Training:** Seamless vertical and horizontal scaling via Ray, from single-node to multi-GPU Data Distributed Parallel (DDP) across cloud clusters.
- **Hyperparameter Optimization:** Grid Search, Random Search, HyperOpt (Bayesian), Optuna, and BOHB (Bayesian with early stopping).
- **Scheduling:** FIFO and ASHA (Asynchronous Successive Halving) schedulers for aggressive trial pruning.
- **Early Stopping:** Convergence-based stopping with configurable patience, minimum delta, and grace period.
- **Learning Rate Scheduling:** StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, and more.
- **Dashboarding:** TensorBoard, Weights & Biases, MLflow, and CodeCarbon for real-time observability.

Evaluation
----------

The Evaluation module provides a suite of **40 multidimensional metrics** organized into 8 functional families:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Family
     - Count
     - Metrics
   * - Accuracy
     - 11
     - nDCG, Precision, Recall, HitRate, MRR, MAP, MAR, F1, AUC, LAUC, GAUC
   * - Rating
     - 3
     - MAE, MSE, RMSE
   * - Coverage
     - 4
     - ItemCoverage, UserCoverage, UserCoverageAtN, NumRetrieved
   * - Novelty
     - 2
     - EPC, EFD
   * - Diversity
     - 3
     - Gini, ShannonEntropy, SRecall
   * - Bias
     - 5
     - ARP, APLT, ACLT, PopREO, PopRSP
   * - Fairness
     - 10
     - REO, RSP, UserMADRanking, UserMADRating, ItemMADRanking, ItemMADRating, BiasDisparityBD, BiasDisparityBS, BiasDisparityBR
   * - Multi-objective
     - 2
     - Hypervolume, EucDistance

Metric computation is **fully GPU-accelerated** using a batch-oriented, tensor-based architecture. A single-pass strategy computes all configured metrics concurrently, eliminating redundant iterations over user data.

WarpRec is the sole framework to natively support **multi-objective metrics** (Hypervolume, Euclidean Distance), enabling model selection through the joint optimization of competing goals such as accuracy and popularity bias.

**Statistical Hypothesis Testing.**
The evaluation pipeline automates significance testing with support for:

- **Paired tests:** Student's t-test, Wilcoxon signed-rank test
- **Independent tests:** Kruskal-Wallis H-test, Mann-Whitney U test
- **Multiple comparison corrections:** Bonferroni, Holm-Bonferroni, FDR (Benjamini-Hochberg)

Writer Module
-------------

The Writer Module ensures reproducibility via a storage-agnostic interface that persists artifacts to local or cloud backends. Beyond standard performance tables, WarpRec automatically serializes:

- Granular per-user metric scores
- Optimized hyperparameters per model
- Trained model weights (checkpoints)
- Recommendation lists
- Execution metadata and timing reports
- Carbon-emission estimates (when CodeCarbon is enabled)

Application Layer
-----------------

WarpRec bridges the gap between experimentation and production deployment through two serving interfaces.

**REST API.**
Trained models can be exposed via a `FastAPI <https://fastapi.tiangolo.com/>`_-powered REST server (``infer-api/server.py``). The API provides versioned endpoints for three recommendation paradigms:

- ``/api/warprec/v1/sequential/`` — Sequential model inference
- ``/api/warprec/v1/collaborative/`` — Collaborative filtering inference
- ``/api/warprec/v1/contextual/`` — Context-aware model inference

**MCP Server.**
The `FastMCP <https://github.com/jlowin/fastmcp>`_-powered MCP server (``infer-api/mcp_server.py``) exposes trained models as callable tools for LLMs and autonomous agents. For example, an LLM can invoke ``recommend_movielens_sequential`` to request personalized recommendations based on a user's viewing history, then reason over the results to generate natural-language explanations.

-----

Pipelines and Callbacks
========================

Pipelines
---------

WarpRec abstracts complex workflows into three standardized execution pipelines, all controlled via declarative YAML configuration files.
For a detailed guide on each pipeline with complete configuration examples and step-by-step walkthroughs, see :doc:`/pipelines/index`.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Pipeline
     - Command
     - Description
   * - **Training**
     - ``-p train``
     - Full end-to-end process: data ingestion, splitting, hyperparameter optimization, training, evaluation, and result persistence.
   * - **Design**
     - ``-p design``
     - Optimized for rapid prototyping and architectural validation. Runs without HPO, requiring single-value hyperparameters.
   * - **Evaluation**
     - ``-p eval``
     - Dedicated to post-hoc analysis and inference using pre-trained checkpoints, without retraining.

Callbacks
---------

The framework integrates an event-driven Callback system based on ``WarpRecCallback``, which extends ``ray.tune.Callback``. Custom callbacks can hook into specific lifecycle stages:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Hook
     - Description
   * - ``on_data_reading(data)``
     - Called after data reading. Can transform the raw DataFrame before processing.
   * - ``on_dataset_creation(main_dataset, val_dataset, validation_folds)``
     - Called after dataset creation. Receives train/test and train/val splits with cross-validation folds.
   * - ``on_training_complete(model)``
     - Called after training finishes. Receives the trained model instance.
   * - ``on_evaluation_complete(model, params, results)``
     - Called after evaluation. Receives the model, hyperparameters, and evaluation results.

Additionally, ``WarpRecCallback`` inherits all `Ray Tune Callback hooks <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Callback.html>`_ such as ``on_trial_start``, ``on_trial_save``, and ``on_trial_complete``.

Callbacks are registered via the configuration file:

.. code-block:: yaml

    general:
        callback:
            callback_path: callbacks/my_callback.py
            callback_name: MyCustomCallback
            kwargs:
                output_dir: ./plots/
