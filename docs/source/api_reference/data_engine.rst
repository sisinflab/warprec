.. _api_data_engine:

######################
Data Engine Reference
######################

The Data Engine manages the full data lifecycle — from ingesting raw transactions to producing model-ready structures. It is composed of four modules: Reader, Filtering, Splitting, and Dataset.

.. note::

    The following API documentation is auto-generated from the WarpRec source code using Sphinx autodoc.

For quick-reference taxonomy tables, see :ref:`Configuration <configuration>`.
For configuration syntax, see :ref:`Configuration <configuration>`.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Reader Module
=============

The Reader module ingests user-item interactions and metadata, abstracting the complexity of data retrieval through a **backend-agnostic** design powered by `Narwhals <https://narwhals-dev.github.io/narwhals/>`_.

Backend Abstraction (Narwhals)
------------------------------

Instead of enforcing a specific DataFrame library, WarpRec uses Narwhals as a compatibility layer. This means:

- **Polars** (default) — optimized for speed and memory efficiency with lazy evaluation.
- **Pandas** — for compatibility with legacy workflows.

The abstraction is transparent: all downstream modules (filtering, splitting, dataset creation) work identically regardless of the backend. No conversion overhead is incurred.

Class Hierarchy
---------------

.. code-block:: text

    Reader (ABC)
    ├── LocalReader       — Reads from local filesystem
    └── AzureBlobReader   — Reads from Azure Blob Storage

**ReaderFactory** dispatches based on the ``reading_method`` configuration key.

.. automodule:: warprec.data.reader.base_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: warprec.data.reader.local_reader
   :members:
   :undoc-members:
   :show-inheritance:

Reader API
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``read_tabular(path, columns, dtypes, sep, header)``
     - Read a CSV/TSV file with optional column selection and type casting.
   * - ``read_parquet(path, columns)``
     - Read a Parquet file with optional column selection.
   * - ``read_tabular_split(path, ...)``
     - Read pre-split train/val/test files (supports multi-fold with numbered subdirectories).
   * - ``read_parquet_split(path, ...)``
     - Same as above for Parquet format.
   * - ``load_model_state(path)``
     - Load a serialized model state (joblib format).

Configuration
-------------

.. code-block:: yaml

    reader:
        loading_strategy: dataset    # "dataset" or "split"
        data_type: transaction       # "transaction"
        reading_method: local        # "local" or "azure_blob"
        local_path: data/movielens.tsv
        rating_type: implicit        # "implicit" or "explicit"
        sep: "\t"                    # Column separator
        labels:
            user_id_label: user_id
            item_id_label: item_id
            rating_label: rating
            timestamp_label: timestamp

For Azure Blob Storage:

.. code-block:: yaml

    reader:
        reading_method: azure_blob
        blob_path: data/movielens.tsv
        storage_account_name: mystorageaccount
        container_name: mycontainer

-----

Filtering Module
================

The Filtering module prunes interactions, users, or items to ensure data quality. Filters are applied **sequentially** in the order specified.

Base Class
----------

.. code-block:: text

    Filter (ABC)
        __call__(dataset: DataFrame) -> DataFrame

All filters accept ``user_id_label``, ``item_id_label``, ``rating_label``, and ``timestamp_label`` as constructor parameters.

.. automodule:: warprec.data.filtering
   :members:
   :undoc-members:
   :show-inheritance:

Strategies (13)
---------------

**By Rating (3)**

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Filter
     - Parameters
     - Description
   * - MinRating
     - ``min_rating: float``
     - Keep interactions with rating :math:`\geq` ``min_rating``.
   * - UserAverage
     - —
     - Keep interactions where rating exceeds the user's average.
   * - ItemAverage
     - —
     - Keep interactions where rating exceeds the item's average.

**K-Core (2)**

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Filter
     - Parameters
     - Description
   * - IterativeKCore
     - ``min_interactions: int``
     - Iteratively remove users/items with fewer than *k* interactions until convergence.
   * - NRoundsKCore
     - ``rounds: int``, ``min_interactions: int``
     - Apply k-core decomposition for exactly *N* rounds.

**Cold-Start (4)**

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Filter
     - Parameters
     - Description
   * - UserMin
     - ``min_interactions: int``
     - Keep users with at least ``min_interactions``.
   * - UserMax
     - ``max_interactions: int``
     - Keep users with at most ``max_interactions``.
   * - ItemMin
     - ``min_interactions: int``
     - Keep items with at least ``min_interactions``.
   * - ItemMax
     - ``max_interactions: int``
     - Keep items with at most ``max_interactions``.

**Temporal (2)**

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Filter
     - Parameters
     - Description
   * - UserHeadN
     - ``num_interactions: int``
     - Keep the first *N* interactions per user (by timestamp).
   * - UserTailN
     - ``num_interactions: int``
     - Keep the last *N* interactions per user (by timestamp).

**Entity Removal (2)**

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Filter
     - Parameters
     - Description
   * - DropUser
     - ``user_ids_to_filter``
     - Exclude specific user(s) by ID.
   * - DropItem
     - ``item_ids_to_filter``
     - Exclude specific item(s) by ID.

Configuration
-------------

.. code-block:: yaml

    filtering:
        - strategy: iterative_k_core
          min_interactions: 5
        - strategy: user_min
          min_interactions: 10

-----

Splitting Module
================

The Splitter partitions filtered data into train, validation, and test sets. All stochastic strategies are anchored to global random seeds for **deterministic reproducibility**.

Base Class
----------

.. code-block:: text

    SplittingStrategy (ABC)
        __call__(data: DataFrame) -> List[Tuple[DataFrame, DataFrame]]

.. automodule:: warprec.data.splitting.strategies
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: warprec.data.splitting.splitter
   :members:
   :undoc-members:
   :show-inheritance:

Strategies (6)
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Strategy
     - Parameters
     - Description
   * - TemporalHoldout
     - ``ratio: float = 0.2``
     - Most recent ``ratio`` fraction becomes test set. Deterministic by timestamp.
   * - TemporalLeaveKOut
     - ``k: int = 1``
     - Last *K* interactions per user (by time) form the test set.
   * - TimestampSlicing
     - ``timestamp: int|str``
     - All interactions after a fixed timestamp form the test set. Accepts ``"best"`` for adaptive split.
   * - RandomHoldout
     - ``ratio: float = 0.2``, ``seed: int = 42``
     - Random split by ratio with fixed seed.
   * - RandomLeaveKOut
     - ``k: int = 1``, ``seed: int = 42``
     - Randomly selects *K* interactions per user for the test set.
   * - KFoldCrossValidation
     - ``folds: int``
     - Creates *N* equal-sized folds per user for cross-validation.

Two-Level Splitting
-------------------

WarpRec supports **nested splitting** for validation:

1. The **test split** is applied first to separate test data.
2. The **validation split** is applied to the remaining data.

This ensures that validation and test sets are disjoint and that no data leaks between them.

Configuration
-------------

.. code-block:: yaml

    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
        validation_splitting:
            strategy: temporal_holdout
            ratio: 0.1

-----

Dataset Module
==============

The Dataset module transforms raw partitions into high-performance internal representations optimized for training and evaluation.

.. automodule:: warprec.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: warprec.data.entities.interactions
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: warprec.data.entities.sessions
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Class
-------------

The central ``Dataset`` class unifies three data types:

- **Interactions** — Sparse user-item interaction matrix (CSR format) for collaborative filtering models.
- **Sessions** — Variable-length user interaction sequences for sequential models.
- **Context** — Auxiliary metadata features for context-aware models.

.. list-table:: Dataset Constructor Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``train_data``
     - DataFrame
     - Training transactions.
   * - ``eval_data``
     - DataFrame (opt.)
     - Evaluation transactions.
   * - ``side_data``
     - DataFrame (opt.)
     - Item feature metadata.
   * - ``user_cluster``
     - DataFrame (opt.)
     - User cluster assignments (for fairness metrics).
   * - ``item_cluster``
     - DataFrame (opt.)
     - Item cluster assignments (for fairness metrics).
   * - ``batch_size``
     - int
     - Training batch size (default: 1024).
   * - ``rating_type``
     - RatingType
     - ``IMPLICIT`` or ``EXPLICIT``.
   * - ``context_labels``
     - List[str] (opt.)
     - Names of contextual feature columns.

Key Methods
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``get_dims() -> (n_users, n_items)``
     - Number of unique users and items.
   * - ``get_mappings() -> (user_map, item_map)``
     - External ID to contiguous internal index mappings.
   * - ``get_inverse_mappings()``
     - Internal index to external ID mappings.
   * - ``get_feature_dims() -> dict``
     - Vocabulary sizes for each side feature.
   * - ``get_context_dims() -> dict``
     - Vocabulary sizes for each context feature.
   * - ``get_features_lookup() -> Tensor``
     - Side information lookup table.
   * - ``get_user_cluster() -> Tensor``
     - User cluster tensor for fairness metrics.
   * - ``get_item_cluster() -> Tensor``
     - Item cluster tensor for fairness metrics.
   * - ``get_evaluation_dataloader()``
     - Full evaluation dataloader (all items).
   * - ``get_sampled_evaluation_dataloader(num_negatives)``
     - Sampled evaluation with negative items.

Data Alignment
--------------

The Dataset module enforces **strict data alignment** by:

1. Mapping external identifiers (user IDs, item IDs) to **contiguous internal indices** starting from 0.
2. Ensuring referential integrity between training and evaluation sets.
3. Materializing optimized sparse matrices (CSR) and padded sequences for high-throughput ingestion.

-----

Writer Module
=============

The Writer module persists all experiment artifacts to local or cloud storage through a **storage-agnostic interface**.

Class Hierarchy
---------------

.. code-block:: text

    Writer (ABC)
    ├── LocalWriter       — Writes to local filesystem
    └── AzureBlobWriter   — Writes to Azure Blob Storage

**WriterFactory** dispatches based on the ``writing_method`` configuration key.

Artifacts Written
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Artifact
     - Description
   * - Performance tables
     - Aggregate metrics across models and cutoffs.
   * - Per-user metrics
     - Granular per-user metric vectors for statistical testing.
   * - Model weights
     - Trained model checkpoints (``.pth`` format).
   * - Hyperparameters
     - Optimized hyperparameters (JSON format).
   * - Recommendation lists
     - Generated recommendation lists per user.
   * - Data splits
     - Serialized train/val/test partitions (CSV/TSV or Parquet).
   * - Time reports
     - Execution time per pipeline stage.
   * - Statistical tests
     - Significance test results with corrections.
   * - Carbon reports
     - CodeCarbon emission data (if enabled).

Configuration
-------------

.. code-block:: yaml

    writer:
        dataset_name: MyExperiment
        writing_method: local                # "local" or "azure_blob"
        local_experiment_path: experiments/

For Azure Blob Storage:

.. code-block:: yaml

    writer:
        dataset_name: MyExperiment
        writing_method: azure_blob
        storage_account_name: mystorageaccount
        container_name: mycontainer
        blob_experiment_container: experiments/
