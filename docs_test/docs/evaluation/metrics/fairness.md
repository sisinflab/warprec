##########
Fairness
##########

.. py:module:: warprec.evaluation.metrics.fairness

Summary
~~~~~~~

.. autosummary::

    reo.REO
    rsp.RSP
    biasdisparitybd.BiasDisparityBD
    biasdisparitybr.BiasDisparityBR
    biasdisparitybs.BiasDisparityBS
    itemmadranking.ItemMADRanking
    itemmadrating.ItemMADRating
    usermadranking.UserMADRanking
    usermadrating.UserMADRating

**Fairness metrics** aim to ensure that recommender systems provide **equitable recommendations** across different user groups, particularly those defined by sensitive attributes (e.g., gender, age, socioeconomic status). These metrics help detect and mitigate **disparate impact or treatment** in recommendation outcomes.

BiasDisparityBD
===============

.. module:: warprec.evaluation.metrics.fairness.biasdisparitybd
.. autoclass:: BiasDisparityBD
    :members:
    :undoc-members:
    :show-inheritance:

Measures the **difference in recommendation bias** between user groups, indicating how much one group is favored over another.

.. math::

    \text{BD}(u, c) = \frac{\text{BR}(u, c) - \text{BS}(u, c)}{\text{BS}(u, c)}

where :math:`\text{BR}` is the bias in recommendations and :math:`\text{BS}` is the bias in the training source. Positive values indicate bias amplification; negative values indicate bias reduction.

**Note:** This metric requires the user to provide clustering information (i.e., user group definitions).

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBD]

BiasDisparityBR (Bias Disparity -- Bias Recommendations)
========================================================

.. module:: warprec.evaluation.metrics.fairness.biasdisparitybr
.. autoclass:: BiasDisparityBR
    :members:
    :undoc-members:
    :show-inheritance:

Quantifies the **disparity in the frequency of biased** (e.g., popular) items recommended to different user groups within their top-K recommendations.

.. math::

    \text{BR}(u, c) = \frac{P_{\text{rec}}(u, c) / P_{\text{rec}}(u)}{P_{\text{global}}(c)}

where :math:`P_{\text{rec}}(u, c)` is the proportion of items from cluster :math:`c` recommended to user cluster :math:`u`, and :math:`P_{\text{global}}(c)` is the global proportion of items in cluster :math:`c`.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBR]

BiasDisparityBS (Bias Disparity -- Bias Scores)
================================================

.. module:: warprec.evaluation.metrics.fairness.biasdisparitybs
.. autoclass:: BiasDisparityBS
    :members:
    :undoc-members:
    :show-inheritance:

Measures the **disparity in the average bias scores** of recommended items across user groups, assessing score-level bias.

.. math::

    \text{BS}(u, c) = \frac{P_{\text{train}}(u, c)}{P_{\text{global}}(c)}

where :math:`P_{\text{train}}(u, c)` is the proportion of interactions from user cluster :math:`u` with items in cluster :math:`c` in the training set.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBS]

Item MAD Ranking
================

.. module:: warprec.evaluation.metrics.fairness.itemmadranking
.. autoclass:: ItemMADRanking
    :members:
    :undoc-members:
    :show-inheritance:

Computes the **Mean Absolute Deviation of item ranks** across user groups, measuring fairness in item exposure in rankings.

.. math::

    \text{ItemMADRanking} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{g}_i - \bar{g}_j \right|

where :math:`\bar{g}_c` is the average discounted gain (DCG) for items in cluster :math:`c` and :math:`m` is the number of item clusters.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRanking]

Item MAD Rating
===============

.. module:: warprec.evaluation.metrics.fairness.itemmadrating
.. autoclass:: ItemMADRating
    :members:
    :undoc-members:
    :show-inheritance:

Computes the **Mean Absolute Deviation of predicted item ratings** across user groups, assessing fairness in predicted preferences.

.. math::

    \text{ItemMADRating} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{r}_i - \bar{r}_j \right|

where :math:`\bar{r}_c` is the average rating/score for items in cluster :math:`c`.

**Note:** This metric requires the user to provide clustering information.

For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRating]

REO (Ranking-based Equal Opportunity)
=====================================

.. module:: warprec.evaluation.metrics.fairness.reo
.. autoclass:: REO
    :members:
    :undoc-members:
    :show-inheritance:

Assesses whether relevant items are **ranked similarly across user groups**, ensuring fair visibility of relevant content.

.. math::

    \text{REO} = \frac{\sigma\bigl(P(R@K \mid g_1, y{=}1), \ldots, P(R@K \mid g_A, y{=}1)\bigr)}{\mu\bigl(P(R@K \mid g_1, y{=}1), \ldots, P(R@K \mid g_A, y{=}1)\bigr)}

where :math:`P(R@K \mid g_a, y{=}1)` is the probability that a relevant item from group :math:`g_a` appears in the top-K recommendations.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [REO]

RSP (Ranking-based Statistical Parity)
======================================

.. module:: warprec.evaluation.metrics.fairness.rsp
.. autoclass:: RSP
    :members:
    :undoc-members:
    :show-inheritance:

Measures whether the **ranking positions of items** (regardless of relevance) are **equally distributed across user groups**, ensuring fairness in exposure.

.. math::

    \text{RSP} = \frac{\sigma\bigl(P(R@K \mid g_1), \ldots, P(R@K \mid g_A)\bigr)}{\mu\bigl(P(R@K \mid g_1), \ldots, P(R@K \mid g_A)\bigr)}

where :math:`P(R@K \mid g_a)` is the probability that an item from group :math:`g_a` is recommended in top-K.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [RSP]

User MAD Ranking
================

.. module:: warprec.evaluation.metrics.fairness.usermadranking
.. autoclass:: UserMADRanking
    :members:
    :undoc-members:
    :show-inheritance:

Measures the **Mean Absolute Deviation of item ranking positions** for each user group, focusing on rank consistency across users.

.. math::

    \text{UserMADRanking} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \text{nDCG}_i - \text{nDCG}_j \right|

where :math:`\text{nDCG}_c` is the average nDCG score for users in cluster :math:`c`.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRanking]

User MAD Rating
===============

.. module:: warprec.evaluation.metrics.fairness.usermadrating
.. autoclass:: UserMADRating
    :members:
    :undoc-members:
    :show-inheritance:

Measures the **Mean Absolute Deviation of predicted item ratings** for each user group, capturing disparities in predicted relevance.

.. math::

    \text{UserMADRating} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{s}_i - \bar{s}_j \right|

where :math:`\bar{s}_c` is the average of top-K recommendation scores for users in cluster :math:`c`.

**Note:** This metric requires the user to provide clustering information.

For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRating]
