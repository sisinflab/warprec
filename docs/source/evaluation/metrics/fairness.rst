##########
Fairness
##########

**Fairness metrics** aim to ensure that recommender systems provide **equitable recommendations** across different user groups, particularly those defined by sensitive attributes (e.g., gender, age, socioeconomic status). These metrics help detect and mitigate **disparate impact or treatment** in recommendation outcomes.

BiasDisparityBD
===============

Measures the **difference in recommendation bias** between user groups, indicating how much one group is favored over another.

**Note:** This metric requires the user to provide clustering information (i.e., user group definitions).

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBD]

BiasDisparityBR (Bias Disparity – Bias Recommendations)
======================================================

Quantifies the **disparity in the frequency of biased** (e.g., popular) items recommended to different user groups within their top-K recommendations.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBR]

BiasDisparityBS (Bias Disparity – Bias Scores)
==============================================

Measures the **disparity in the average bias scores** of recommended items across user groups, assessing score-level bias.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [BiasDisparityBS]

Item MAD Ranking
================

Computes the **Mean Absolute Deviation of item ranks** across user groups, measuring fairness in item exposure in rankings.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRanking]

Item MAD Rating
===============

Computes the **Mean Absolute Deviation of predicted item ratings** across user groups, assessing fairness in predicted preferences.

**Note:** This metric requires the user to provide clustering information.

For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemMADRating]

REO (Ranking-based Equal Opportunity)
=====================================

Assesses whether relevant items are **ranked similarly across user groups**, ensuring fair visibility of relevant content.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [REO]

RSP (Ranking-based Statistical Parity)
======================================

Measures whether the **ranking positions of items** (regardless of relevance) are **equally distributed across user groups**, ensuring fairness in exposure.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [RSP]

User MAD Ranking
================

Measures the **Mean Absolute Deviation of item ranking positions** for each user group, focusing on rank consistency across users.

**Note:** This metric requires the user to provide clustering information.

For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRanking]

User MAD Rating
===============

Measures the **Mean Absolute Deviation of predicted item ratings** for each user group, capturing disparities in predicted relevance.

**Note:** This metric requires the user to provide clustering information.

For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserMADRating]
