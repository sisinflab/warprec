.. _api_metrics:

##################
Metrics Reference
##################

WarpRec provides 40 GPU-accelerated evaluation metrics implemented as PyTorch modules.
All metrics extend ``BaseMetric`` via TorchMetrics and support distributed evaluation.

.. autoclass:: warprec.evaluation.metrics.base_metric.BaseMetric
    :members:
    :show-inheritance:

For quick-reference taxonomy tables, see :ref:`Evaluation <evaluation>`.
For configuration syntax, see :ref:`Configuration <configuration>`.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Accuracy Metrics
================

Accuracy metrics quantify how well a recommender identifies relevant items.

Precision@K
-----------

Proportion of recommended items at rank K that are relevant.

.. math::

    \text{Precision@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{K}

where :math:`\mathcal{R}_u` is the set of relevant items for user :math:`u` and :math:`\mathcal{L}_u^K` is the top-K recommendation list.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Precision]

Recall@K
--------

Proportion of relevant items that are successfully retrieved within the top K.

.. math::

    \text{Recall@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{|\mathcal{R}_u|}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Recall]

nDCG@K
------

Normalized Discounted Cumulative Gain evaluates ranking quality with exponential relevance gain.

.. math::

    \text{DCG@}K = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}, \quad
    \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}

where :math:`r_i` is the relevance of the item at rank :math:`i`, and IDCG@K is the ideal DCG computed from the perfect ranking.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG]

nDCGRendle2020@K
~~~~~~~~~~~~~~~~

A variant that uses binary relevance, following `Rendle et al. (2020) <https://dl.acm.org/doi/10.1145/3394486.3403226>`_:

.. math::

    \text{DCG@}K = \sum_{i=1}^{K} \frac{r_i}{\log_2(i + 1)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCGRendle2020]

HitRate@K
---------

Fraction of users for whom at least one relevant item appears in the top K.

.. math::

    \text{HitRate@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}\left[|\mathcal{R}_u \cap \mathcal{L}_u^K| > 0\right]

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [HitRate]

MAP@K
-----

Mean Average Precision rewards correct recommendations ranked higher.

.. math::

    \text{AP@}K(u) = \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Precision@}i \cdot r_i, \quad
    \text{MAP@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{AP@}K(u)

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAP]

MAR@K
-----

Mean Average Recall measures progressive retrieval quality.

.. math::

    \text{AR@}K(u) = \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Recall@}i \cdot r_i, \quad
    \text{MAR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{AR@}K(u)

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MAR]

MRR@K
-----

Mean Reciprocal Rank averages the reciprocal of the rank of the first relevant item.

.. math::

    \text{MRR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u}

where :math:`\text{rank}_u` is the position of the first relevant item in the top-K list (or 0 if none).

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [MRR]

F1@K
----

Harmonic mean of Precision@K and Recall@K (or any two metrics).

.. math::

    \text{F1@}K = \frac{(1 + \beta^2) \cdot m_1 \cdot m_2}{\beta^2 \cdot m_1 + m_2}

With default :math:`\beta = 1`, :math:`m_1 = \text{Precision@}K`, and :math:`m_2 = \text{Recall@}K`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [F1]

**Extended F1** allows custom metric pairs:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: F1
              params:
                  metric_name_1: nDCG
                  metric_name_2: MAP
                  beta: 0.5

AUC
---

Area Under the ROC Curve measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.

.. math::

    \text{AUC} = \frac{1}{|\mathcal{P}|} \sum_{(u,i) \in \mathcal{P}} \frac{|\{j \in \mathcal{N}_u : \hat{r}_{ui} > \hat{r}_{uj}\}|}{|\mathcal{N}_u|}

where :math:`\mathcal{P}` is the set of positive (user, item) pairs and :math:`\mathcal{N}_u` is the set of negative items for user :math:`u`.

.. code-block:: yaml

    evaluation:
        metrics: [AUC]

GAUC
----

Group AUC computes AUC per user and averages the results, accounting for group-level ranking quality.

.. math::

    \text{GAUC} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{AUC}_u

.. code-block:: yaml

    evaluation:
        metrics: [GAUC]

LAUC@K
------

Limited AUC focuses on ranking quality within the top-K positions.

.. math::

    \text{LAUC@}K = \frac{\text{AUC computed over top-}K\text{ items}}{\min(|\mathcal{R}_u|, K)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [LAUC]

-----

Rating Metrics
==============

Rating metrics measure the accuracy of explicit rating predictions.

MAE
---

Mean Absolute Error — the average absolute difference between predicted and actual ratings.

.. math::

    \text{MAE} = \frac{1}{N} \sum_{(u,i)} |\hat{r}_{ui} - r_{ui}|

.. code-block:: yaml

    evaluation:
        metrics: [MAE]

MSE
---

Mean Squared Error — the average of squared differences.

.. math::

    \text{MSE} = \frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2

.. code-block:: yaml

    evaluation:
        metrics: [MSE]

RMSE
----

Root Mean Squared Error — provides error in the same units as ratings.

.. math::

    \text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2}

.. code-block:: yaml

    evaluation:
        metrics: [RMSE]

-----

Coverage Metrics
================

Coverage metrics assess how broadly a recommender explores the item catalog.

ItemCoverage@K
--------------

Number of unique items recommended across all users.

.. math::

    \text{ItemCoverage@}K = \left|\bigcup_{u \in \mathcal{U}} \mathcal{L}_u^K\right|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemCoverage]

UserCoverage@K
--------------

Number of users with at least one recommended item in the top K.

.. math::

    \text{UserCoverage@}K = \left|\{u \in \mathcal{U} : |\mathcal{L}_u^K| > 0\}\right|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverage]

NumRetrieved@K
--------------

Average number of items retrieved per user in the top K.

.. math::

    \text{NumRetrieved@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \min(K, |\mathcal{L}_u|)

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [NumRetrieved]

UserCoverageAtN
---------------

Number of users that retrieved at least N items.

.. math::

    \text{UserCoverageAtN} = |\{u \in \mathcal{U} : |\mathcal{L}_u| \geq K\}|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverageAtN]

-----

Novelty Metrics
===============

Novelty metrics assess whether recommendations go beyond popular items.

EFD@K
-----

Expected Free Discovery — system's ability to suggest novel (less popular) items, weighted by logarithmic novelty.

.. math::

    \text{EFD@}K = \frac{1}{|\mathcal{U}| \cdot C} \sum_{u \in \mathcal{U}} \sum_{i=1}^{K} \frac{r_i \cdot (-\log_2 p_i)}{\log_2(i + 1)}

where :math:`p_i = n_i / |\mathcal{U}|` is the popularity of item :math:`i` (interaction count normalized by number of users) and :math:`C = \sum_{i=1}^{K} 1/\log_2(i+1)`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [EFD]

**Extended EFD** with discounted relevance:

.. code-block:: yaml

    evaluation:
        complex_metrics:
            - name: EFD
              params:
                  relevance: discounted

EPC@K
-----

Expected Popularity Complement — similar to EFD but with linear novelty.

.. math::

    \text{EPC@}K = \frac{1}{|\mathcal{U}| \cdot C} \sum_{u \in \mathcal{U}} \sum_{i=1}^{K} \frac{r_i \cdot (1 - p_i)}{\log_2(i + 1)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [EPC]

-----

Diversity Metrics
=================

Diversity metrics evaluate the variety of items within recommendations.

Gini Index@K
-------------

Measures inequality in the distribution of recommended items. Lower values indicate more equitable exposure.

.. math::

    \text{Gini} = \frac{\sum_{j=1}^{|\mathcal{I}|} (2j - |\mathcal{I}| - 1) \cdot c_j}{(|\mathcal{I}| - 1) \cdot \sum_{j=1}^{|\mathcal{I}|} c_j}

where :math:`c_j` is the recommendation count for item :math:`j` (sorted in ascending order).

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Gini]

Shannon Entropy@K
-----------------

Information entropy over item recommendation frequencies. Higher values reflect greater item variety.

.. math::

    H = -\sum_{i \in \mathcal{I}} p_i \log p_i, \quad p_i = \frac{c_i}{\sum_{j} c_j}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ShannonEntropy]

SRecall@K
---------

Subtopic Recall measures how many distinct feature categories are covered in the top-K recommendations.

.. math::

    \text{SRecall@}K(u) = \frac{|\text{features}(\mathcal{L}_u^K) \cap \text{features}(\mathcal{R}_u)|}{|\text{features}(\mathcal{R}_u)|}

.. note::

    This metric requires side information (item categories/features).

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [SRecall]

-----

Bias Metrics
============

Bias metrics identify systematic deviations in recommendation outputs, particularly regarding the popularity of recommended items.

.. note::

    Bias metrics that reference short-head/long-tail use a configurable ``pop_ratio`` (default: 0.8). Items in the top ``pop_ratio`` fraction by popularity are "short-head"; the rest are "long-tail."

ARP@K
-----

Average Recommendation Popularity — the mean popularity of recommended items.

.. math::

    \text{ARP@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{K} \sum_{i \in \mathcal{L}_u^K} \phi(i)

where :math:`\phi(i)` is the interaction count (popularity) of item :math:`i`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ARP]

ACLT@K
------

Average Coverage of Long-Tail items in the top-K recommendations.

.. math::

    \text{ACLT@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} |\{i \in \mathcal{L}_u^K : i \in \mathcal{T}\}|

where :math:`\mathcal{T}` is the set of long-tail items.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ACLT]

APLT@K
------

Average Proportion of Long-Tail items per user.

.. math::

    \text{APLT@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\{i \in \mathcal{L}_u^K : i \in \mathcal{T}\}|}{K}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [APLT]

PopRSP@K
--------

Popularity-based Ranking-based Statistical Parity — disparity in recommendation performance between short-head and long-tail groups.

.. math::

    \text{PopRSP@}K = \frac{\text{std}(P_\text{short}, P_\text{long})}{\text{mean}(P_\text{short}, P_\text{long})}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopRSP]

PopREO@K
--------

Popularity-based Ranking-based Equal Opportunity — fairness comparing short-head vs long-tail item recommendations conditioned on relevance.

.. math::

    \text{PopREO@}K = \frac{\text{std}(P_\text{short}^+, P_\text{long}^+)}{\text{mean}(P_\text{short}^+, P_\text{long}^+)}

where :math:`P_g^+` denotes the recommendation rate for relevant items in group :math:`g`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopREO]

Custom ``pop_ratio`` for any bias metric:

.. code-block:: yaml

    evaluation:
        complex_metrics:
            - name: ACLT
              params:
                  pop_ratio: 0.9

-----

Fairness Metrics
================

Fairness metrics ensure equitable recommendations across user or item groups defined by sensitive attributes.

.. note::

    All fairness metrics require **clustering information** — user group and/or item group assignments — provided via configuration.

REO@K
-----

Ranking-based Equal Opportunity — whether relevant items are ranked similarly across item clusters.

.. math::

    \text{REO@}K = \frac{\text{std}(\{P(R@K | g = g_a, y = 1)\}_{a})}{\text{mean}(\{P(R@K | g = g_a, y = 1)\}_{a})}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [REO]

RSP@K
-----

Ranking-based Statistical Parity — whether item exposure is equally distributed across item clusters.

.. math::

    \text{RSP@}K = \frac{\text{std}(\{P(R@K | g = g_a)\}_{a})}{\text{mean}(\{P(R@K | g = g_a)\}_{a})}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [RSP]

BiasDisparityBD@K
-----------------

Relative disparity between recommendation bias (BR) and training bias (BS).

.. math::

    \text{BD}(u, c) = \frac{\text{BR}(u, c) - \text{BS}(u, c)}{\text{BS}(u, c)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBD]

BiasDisparityBR@K
-----------------

Bias in recommendation frequency per user-item cluster pair.

.. math::

    \text{BR}(u, c) = \frac{P_\text{rec}(u, c) / P_\text{rec}(u)}{P_\text{global}(c)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBR]

BiasDisparityBS
---------------

Bias in the training set per user-item cluster pair.

.. math::

    \text{BS}(u, c) = \frac{P_\text{train}(u, c)}{P_\text{global}(c)}

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBS]

ItemMADRanking@K
----------------

Mean Absolute Deviation of discounted gain across item clusters.

.. math::

    \text{ItemMADRanking@}K = \frac{1}{\binom{|C|}{2}} \sum_{c_1, c_2} |\bar{g}_{c_1} - \bar{g}_{c_2}|

where :math:`\bar{g}_c` is the average discounted gain for items in cluster :math:`c`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRanking]

ItemMADRating@K
---------------

Mean Absolute Deviation of average ratings for relevant items across item clusters.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRating]

UserMADRanking@K
----------------

Mean Absolute Deviation of nDCG scores across user clusters.

.. math::

    \text{UserMADRanking@}K = \frac{1}{\binom{|G|}{2}} \sum_{g_1, g_2} |\overline{\text{nDCG}}_{g_1} - \overline{\text{nDCG}}_{g_2}|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRanking]

UserMADRating@K
---------------

Mean Absolute Deviation of average top-K scores across user clusters.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRating]

-----

Multi-objective Metrics
=======================

Multi-objective metrics evaluate performance across multiple, often conflicting, objectives simultaneously.

Euclidean Distance@K
--------------------

Distance between the model's performance and an ideal Utopia Point. **Lower is better.**

.. math::

    d = \sqrt{\sum_{j=1}^{M} (s_j - u_j)^2}

where :math:`s_j` is the model's score on metric :math:`j` and :math:`u_j` is the utopia (ideal) value.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: EucDistance
              params:
                  metric_names: [Precision, Recall, nDCG]
                  utopia_points: [1.0, 1.0, 1.0]

Hypervolume@K
-------------

Volume of the objective space dominated by the model relative to a Nadir Point. **Higher is better.**

.. math::

    \text{HV} = \prod_{j=1}^{M} \max(0, s_j - n_j)

where :math:`n_j` is the nadir (worst-case) reference for metric :math:`j`. For metrics where lower is better, the formula uses :math:`\max(0, n_j - s_j)`.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: Hypervolume
              params:
                  metric_names: [Precision, Recall, nDCG]
                  nadir_points: [0.0, 0.0, 0.0]
                  higher_is_better: [True, True, True]
