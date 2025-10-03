#####
Bias
#####

**Bias metrics** are designed to identify and measure systematic deviations or unfair tendencies in recommender system outputs. These metrics help uncover whether the system disproportionately favors or disfavors certain items, users, or groups, potentially leading to a lack of **diversity** or **equitability** in recommendations.

ACLT (Average Coverage of Long-Tail items)
=========================================

Measures the proportion of **long-tail items** recommended across all users, indicating the extent of long-tail exposure.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ACLT]

APLT (Average Proportion of Long-Tail items)
===========================================

Measures the average proportion of **long-tail items** in each user's recommendation list, which captures individual-level diversity.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [APLT]

ARP (Average Recommendation Popularity)
======================================

Calculates the **average popularity** of recommended items, indicating the system’s tendency to favor popular content.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ARP]

PopREO (Popularity-based Ranking-based Equal Opportunity)
=========================================================

Measures whether users receive similar ranks for **long-tail items** regardless of their group membership, focusing on **fairness in exposure**.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopREO]

PopRSP (Popularity-based Ranking-based Statistical Parity)
==========================================================

Evaluates whether the average ranks of **long-tail items** are balanced across user groups, promoting **fairness in recommendation ranking**.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [PopRSP]
