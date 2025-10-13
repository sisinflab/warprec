#########
Accuracy
#########

**Accuracy metrics** quantify how well a recommender system predicts user preferences or identifies relevant items. They assess the **correctness** of recommendations by comparing predicted interactions or ratings against actual user behavior. High accuracy generally indicates that the system is effective at surfacing items users are likely to engage with.

Precision@K
============

Measures the proportion of recommended items at rank K that are actually relevant.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Precision]

Recall@K
========

Measures the proportion of relevant items that are successfully recommended within the top K items.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Recall]

F1-Score@K
==========

The harmonic mean of Precision@K and Recall@K, providing a balanced measure of accuracy.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [F1]

**Extended-F1** is also available, allowing you to compute the harmonic mean of any two metrics of your choice, as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: ["F1[nDCG, MAP]"]

HitRate@K
=========

Measures the percentage of users for whom at least one relevant item is found within the top K recommendations.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [HitRate]

NDCG@K (Normalized Discounted Cumulative Gain)
==============================================

Evaluates the **ranking quality** of recommendations, giving higher scores to relevant items appearing at higher ranks.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG]

**nDCGRendle2020** is also available, allowing you to compute nDCG on binary relevance.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCGRendle2020]

MAP@K (Mean Average Precision)
==============================

Measures the mean of average precision scores across all users, rewarding correct recommendations ranked higher.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAP]

MAR@K (Mean Average Recall)
===========================

Measures the mean of average recall scores across all users, indicating how well the relevant items are retrieved on average.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAR]

MRR@K (Mean Reciprocal Rank)
============================

Measures the average of the reciprocal ranks of the first relevant item in the recommendations.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MRR]

AUC (Area Under the ROC Curve)
==============================

Measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.

.. code-block:: yaml

    evaluation:
        metrics: [AUC]

GAUC (Group Area Under the ROC Curve)
======================================

Computes AUC per user (or group), then averages the results; accounts for group-level ranking quality.

.. code-block:: yaml

    evaluation:
        metrics: [GAUC]

LAUC (Limited Area Under the ROC Curve)
=======================================

AUC computed over a limited set of top-ranked items, focusing on ranking quality within the most relevant recommendations.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [LAUC]
