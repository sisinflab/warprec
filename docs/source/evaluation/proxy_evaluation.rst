.. _proxy-evaluation:

######################################
Evaluating External Recommendations
######################################

WarpRec can evaluate precomputed recommendation files produced by itself or by external frameworks (e.g., Elliot, RecBole, Cornac, DaisyRec). This capability is provided by the ``ProxyRecommender``, a special model that loads an existing recommendation file and feeds it into the evaluation module as if it were a native WarpRec model.

This is a **cross-cutting concern** that applies to all recommendation types and pipelines — not just unpersonalized models. It enables direct, fair comparison of recommendations from any source using WarpRec's 40 GPU-accelerated metrics and statistical significance testing.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

How It Works
============

``ProxyRecommender`` reads a precomputed recommendation file (TSV or CSV) containing user-item-score triples. It constructs an internal sparse matrix from these scores and exposes a ``predict()`` method that returns the precomputed scores for any given user. From the evaluation module's perspective, it behaves identically to any other trained model.

**Source:** ``warprec/recommenders/unpersonalized_recommender/proxy.py``

File Format Requirements
========================

The recommendation file must contain at least two columns:

- **user_id** — The user identifier (must match the dataset's user IDs).
- **item_id** — The item identifier (must match the dataset's item IDs).
- **score** (optional) — A numeric relevance score. If absent, all recommendations are treated as equally relevant.

Example (TSV with header):

.. code-block:: text

    user_id	item_id	score
    1	1193	4.85
    1	661	4.72
    1	914	4.65
    2	1357	4.91
    2	3068	4.83

Configuration Parameters
========================

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``recommendation_file``
     - ``str``
     - Path to the recommendation file.
   * - ``separator``
     - ``str``
     - Column separator (default: ``"\t"``).
   * - ``header``
     - ``bool``
     - Whether the file includes a header row (default: ``true``).

Usage Examples
==============

Example 1: Evaluate a Single External File
-------------------------------------------

.. code-block:: yaml

    models:
        ExternalBaseline:
            meta:
                model_name: ProxyRecommender
            recommendation_file: results/elliot_lightgcn_recs.tsv
            separator: "\t"
            header: true

    evaluation:
        top_k: [10, 20]
        metrics: [nDCG, Precision, Recall, HitRate]

Example 2: Compare External with Native Models
-----------------------------------------------

You can evaluate external recommendations alongside native WarpRec models for direct comparison:

.. code-block:: yaml

    models:
        Elliot_LightGCN:
            meta:
                model_name: ProxyRecommender
            recommendation_file: results/elliot_lightgcn_recs.tsv
            separator: "\t"
            header: true
        RecBole_SASRec:
            meta:
                model_name: ProxyRecommender
            recommendation_file: results/recbole_sasrec_recs.tsv
            separator: "\t"
            header: true
        LightGCN:
            embedding_size: 64
            n_layers: 3
            reg_weight: 0.001
            batch_size: 2048
            epochs: 200
            learning_rate: 0.001

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG, Precision, Recall, HitRate, MAP]
        stat_significance:
            wilcoxon_test: true
            corrections:
                bonferroni: true

Example 3: Using the Evaluation Pipeline
-----------------------------------------

For pure evaluation without training, use the Evaluation Pipeline (``-p eval``):

.. code-block:: yaml

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
        Framework_A:
            meta:
                model_name: ProxyRecommender
            recommendation_file: results/framework_a.tsv
            separator: "\t"
            header: true
        Framework_B:
            meta:
                model_name: ProxyRecommender
            recommendation_file: results/framework_b.tsv
            separator: ","
            header: true

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG, Precision, Recall, HitRate, MAP, MRR, F1, AUC]

.. code-block:: bash

    python -m warprec.run -c config/compare.yml -p eval

Limitations
===========

.. warning::

    ``ProxyRecommender`` must execute **locally**. It will raise an error if distributed Ray resources are configured (e.g., ``gpu_per_trial > 0``). Use it only in the **Design Pipeline** or **Evaluation Pipeline**, not in the Training Pipeline with Ray-based HPO.

.. note::

    The user and item IDs in the recommendation file must match those in the dataset after filtering and splitting. If IDs are mismatched, the evaluation will silently ignore unrecognized entries.

See Also
========

- :doc:`/pipelines/eval` — Evaluation Pipeline documentation.
- :doc:`/recommenders/unpersonalized` — Unpersonalized recommenders (Pop, Random, ProxyRecommender).
- :doc:`/evaluation/index` — Evaluation module overview.
