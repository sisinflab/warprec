#####
Bias
#####

**Bias metrics** are designed to identify and measure systematic deviations or unfair tendencies in recommender system outputs. These metrics help uncover whether the system disproportionately favors or disfavors certain items, users, or groups, potentially leading to a lack of **diversity** or **equitability** in recommendations.

.. note::

    Some metrics support a custom value for the computation of the short-head/long-tail. The default proportion is: 0.8.

ACLT (Average Coverage of Long-Tail items)
=========================================

Measures the proportion of **long-tail items** recommended across all users, indicating the extent of long-tail exposure.

For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ACLT]

You can also compute this metric using a different proportion of popularity for the short-head/long-tail computation as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: ACLT
              params:
                  pop_ratio: 0.9

APLT (Average Proportion of Long-Tail items)
===========================================

Measures the average proportion of **long-tail items** in each user's recommendation list, which captures individual-level diversity.

For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [APLT]

You can also compute this metric using a different proportion of popularity for the short-head/long-tail computation as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: APLT
              params:
                  pop_ratio: 0.9

ARP (Average Recommendation Popularity)
======================================

Calculates the **average popularity** of recommended items, indicating the system’s tendency to favor popular content.

For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ARP]

PopREO (Popularity-based Ranking-based Equal Opportunity)
=========================================================

Measures whether users receive similar ranks for **long-tail items** regardless of their group membership, focusing on **fairness in exposure**.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopREO]

You can also compute this metric using a different proportion of popularity for the short-head/long-tail computation as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: PopREO
              params:
                  pop_ratio: 0.9

PopRSP (Popularity-based Ranking-based Statistical Parity)
==========================================================

Evaluates whether the average ranks of **long-tail items** are balanced across user groups, promoting **fairness in recommendation ranking**.

For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopRSP]

You can also compute this metric using a different proportion of popularity for the short-head/long-tail computation as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: PopRSP
              params:
                  pop_ratio: 0.9
