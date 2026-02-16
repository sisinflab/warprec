#########
Accuracy
#########

**Accuracy metrics** quantify how well a recommender system predicts user preferences or identifies relevant items. They assess the **correctness** of recommendations by comparing predicted interactions or ratings against actual user behavior. High accuracy generally indicates that the system is effective at surfacing items users are likely to engage with.

AUC (Area Under the ROC Curve)
==============================

Measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

.. code-block:: yaml

    evaluation:
        metrics: [AUC]

F1-Score@K
==========

The harmonic mean of Precision@K and Recall@K, providing a balanced measure of accuracy.

For further details, please refer to this `book <https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8>`_ and this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [F1]

**Extended-F1** is also available, allowing you to compute the harmonic mean of any two metrics of your choice, as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: F1
              params:
                  metric_name_1: nDCG
                  metric_name_2: MAP
                  beta: 0.5

GAUC (Group Area Under the ROC Curve)
======================================

Computes AUC per user (or group), then averages the results; accounts for group-level ranking quality.

For further details, please refer to this `paper <https://www.ijcai.org/Proceedings/2019/0319.pdf>`_.

.. code-block:: yaml

    evaluation:
        metrics: [GAUC]

HitRate@K
=========

Measures the percentage of users for whom at least one relevant item is found within the top K recommendations.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Hit_rate>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [HitRate]

LAUC (Limited Area Under the ROC Curve)
=======================================

AUC computed over a limited set of top-ranked items, focusing on ranking quality within the most relevant recommendations.

For further details, please refer to this `paper <https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [LAUC]

MAP@K (Mean Average Precision)
==============================

Measures the mean of average precision scores across all users, rewarding correct recommendations ranked higher.

For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms>`_.v

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAP]

MAR@K (Mean Average Recall)
===========================

Measures the mean of average recall scores across all users, indicating how well the relevant items are retrieved on average.

For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAR]

MRR@K (Mean Reciprocal Rank)
============================

Measures the average of the reciprocal ranks of the first relevant item in the recommendations.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MRR]

NDCG@K (Normalized Discounted Cumulative Gain)
==============================================

Evaluates the **ranking quality** of recommendations, giving higher scores to relevant items appearing at higher ranks.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG]

**nDCGRendle2020** is also available, allowing you to compute nDCG on binary relevance.

For further details, please refer to this `link <https://dl.acm.org/doi/10.1145/3394486.3403226>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCGRendle2020]

Precision@K
============

Measures the proportion of recommended items at rank K that are actually relevant.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Precision]

Recall@K
========

Measures the proportion of relevant items that are successfully recommended within the top K items.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Recall]
