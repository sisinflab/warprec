#########
Accuracy
#########

.. py:module:: warprec.evaluation.metrics.accuracy

Summary
~~~~~~~

.. autosummary::

    auc.AUC
    f1.F1
    gauc.GAUC
    hit_rate.HitRate
    lauc.LAUC
    map.MAP
    mar.MAR
    mrr.MRR
    ndcg.nDCG
    ndcg.nDCGRendle2020
    precision.Precision
    recall.Recall

**Accuracy metrics** quantify how well a recommender system predicts user preferences or identifies relevant items. They assess the **correctness** of recommendations by comparing predicted interactions or ratings against actual user behavior. High accuracy generally indicates that the system is effective at surfacing items users are likely to engage with.

AUC (Area Under the ROC Curve)
==============================

.. module:: warprec.evaluation.metrics.accuracy.auc
.. autoclass:: AUC
    :members:
    :undoc-members:
    :show-inheritance:

Measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.

.. math::

    \text{AUC} = \frac{1}{|\mathcal{P}|} \sum_{(u,i) \in \mathcal{P}} \frac{|\{j \in \mathcal{N}_u : \hat{r}_{ui} > \hat{r}_{uj}\}|}{|\mathcal{N}_u|}

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

.. code-block:: yaml

    evaluation:
        metrics: [AUC]

F1-Score@K
==========

.. module:: warprec.evaluation.metrics.accuracy.f1
.. autoclass:: F1
    :members:
    :undoc-members:
    :show-inheritance:

The harmonic mean of Precision@K and Recall@K, providing a balanced measure of accuracy.

.. math::

    \text{F1@}K = \frac{(1 + \beta^2) \cdot \text{Precision@}K \cdot \text{Recall@}K}{\beta^2 \cdot \text{Precision@}K + \text{Recall@}K}

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

.. module:: warprec.evaluation.metrics.accuracy.gauc
.. autoclass:: GAUC
    :members:
    :undoc-members:
    :show-inheritance:

Computes AUC per user (or group), then averages the results; accounts for group-level ranking quality.

.. math::

    \text{GAUC} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{AUC}_u

For further details, please refer to this `paper <https://www.ijcai.org/Proceedings/2019/0319.pdf>`_.

.. code-block:: yaml

    evaluation:
        metrics: [GAUC]

HitRate@K
=========

.. module:: warprec.evaluation.metrics.accuracy.hit_rate
.. autoclass:: HitRate
    :members:
    :undoc-members:
    :show-inheritance:

Measures the percentage of users for whom at least one relevant item is found within the top K recommendations.

.. math::

    \text{HitRate@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}\left[|\mathcal{R}_u \cap \mathcal{L}_u^K| > 0\right]

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Hit_rate>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [HitRate]

LAUC (Limited Area Under the ROC Curve)
=======================================

.. module:: warprec.evaluation.metrics.accuracy.lauc
.. autoclass:: LAUC
    :members:
    :undoc-members:
    :show-inheritance:

AUC computed over a limited set of top-ranked items, focusing on ranking quality within the most relevant recommendations.

For further details, please refer to this `paper <https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [LAUC]

MAP@K (Mean Average Precision)
==============================

.. module:: warprec.evaluation.metrics.accuracy.map
.. autoclass:: MAP
    :members:
    :undoc-members:
    :show-inheritance:

Measures the mean of average precision scores across all users, rewarding correct recommendations ranked higher.

.. math::

    \text{MAP@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Precision@}i \cdot r_i

For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAP]

MAR@K (Mean Average Recall)
===========================

.. module:: warprec.evaluation.metrics.accuracy.mar
.. autoclass:: MAR
    :members:
    :undoc-members:
    :show-inheritance:

Measures the mean of average recall scores across all users, indicating how well the relevant items are retrieved on average.

.. math::

    \text{MAR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Recall@}i \cdot r_i

For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAR]

MRR@K (Mean Reciprocal Rank)
============================

.. module:: warprec.evaluation.metrics.accuracy.mrr
.. autoclass:: MRR
    :members:
    :undoc-members:
    :show-inheritance:

Measures the average of the reciprocal ranks of the first relevant item in the recommendations.

.. math::

    \text{MRR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u}

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MRR]

NDCG@K (Normalized Discounted Cumulative Gain)
==============================================

.. module:: warprec.evaluation.metrics.accuracy.ndcg
.. autoclass:: nDCG
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nDCGRendle2020
    :members:
    :undoc-members:
    :show-inheritance:

Evaluates the **ranking quality** of recommendations, giving higher scores to relevant items appearing at higher ranks.

.. math::

    \text{DCG@}K = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}

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

.. module:: warprec.evaluation.metrics.accuracy.precision
.. autoclass:: Precision
    :members:
    :undoc-members:
    :show-inheritance:

Measures the proportion of recommended items at rank K that are actually relevant.

.. math::

    \text{Precision@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{K}

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Precision]

Recall@K
========

.. module:: warprec.evaluation.metrics.accuracy.recall
.. autoclass:: Recall
    :members:
    :undoc-members:
    :show-inheritance:

Measures the proportion of relevant items that are successfully recommended within the top K items.

.. math::

    \text{Recall@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{|\mathcal{R}_u|}

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Recall]
