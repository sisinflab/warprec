#####
Bias
#####

.. py:module:: warprec.evaluation.metrics.bias

Summary
~~~~~~~

.. autosummary::

    arp.ARP
    aclt.ACLT
    aplt.APLT
    pop_rsp.PopRSP
    pop_reo.PopREO

**Bias metrics** are designed to identify and measure systematic deviations or unfair tendencies in recommender system outputs. These metrics help uncover whether the system disproportionately favors or disfavors certain items, users, or groups, potentially leading to a lack of **diversity** or **equitability** in recommendations.

.. note::

    Some metrics support a custom value for the computation of the short-head/long-tail. The default proportion is: 0.8.

ACLT (Average Coverage of Long-Tail items)
==========================================

.. module:: warprec.evaluation.metrics.bias.aclt
.. autoclass:: ACLT
    :members:
    :undoc-members:
    :show-inheritance:

Measures the proportion of **long-tail items** recommended across all users, indicating the extent of long-tail exposure.

.. math::

    \text{ACLT@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \left| \mathcal{L}_u^K \cap \mathcal{I}_{\text{long}} \right|

where :math:`\mathcal{I}_{\text{long}}` is the set of long-tail items (determined by the popularity percentile threshold, default 0.8).

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
=============================================

.. module:: warprec.evaluation.metrics.bias.aplt
.. autoclass:: APLT
    :members:
    :undoc-members:
    :show-inheritance:

Measures the average proportion of **long-tail items** in each user's recommendation list, which captures individual-level diversity.

.. math::

    \text{APLT@}K = \frac{1}{|\mathcal{U}| \cdot K} \sum_{u \in \mathcal{U}} \left| \mathcal{L}_u^K \cap \mathcal{I}_{\text{long}} \right|

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
=======================================

.. module:: warprec.evaluation.metrics.bias.arp
.. autoclass:: ARP
    :members:
    :undoc-members:
    :show-inheritance:

Calculates the **average popularity** of recommended items, indicating the system’s tendency to favor popular content.

.. math::

    \text{ARP@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{K} \sum_{i \in \mathcal{L}_u^K} \text{pop}(i)

where :math:`\text{pop}(i)` is the number of interactions for item :math:`i` in the training set.

For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ARP]

PopREO (Popularity-based Ranking-based Equal Opportunity)
=========================================================

.. module:: warprec.evaluation.metrics.bias.pop_reo
.. autoclass:: PopREO
    :members:
    :undoc-members:
    :show-inheritance:

Measures whether users receive similar ranks for **long-tail items** regardless of their group membership, focusing on **fairness in exposure**.

.. math::

    \text{PopREO} = \frac{\sigma(p_{\text{short}},\; p_{\text{long}})}{\mu(p_{\text{short}},\; p_{\text{long}})}, \quad p_g = \frac{|\mathcal{L}^K \cap \mathcal{I}_g|}{|\mathcal{R} \cap \mathcal{I}_g|}

where :math:`p_g` is the ratio of recommended items from group :math:`g` to the number of relevant items from that group.

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

.. module:: warprec.evaluation.metrics.bias.pop_rsp
.. autoclass:: PopRSP
    :members:
    :undoc-members:
    :show-inheritance:

Evaluates whether the average ranks of **long-tail items** are balanced across user groups, promoting **fairness in recommendation ranking**.

.. math::

    \text{PopRSP} = \frac{\sigma(p_{\text{short}},\; p_{\text{long}})}{\mu(p_{\text{short}},\; p_{\text{long}})}, \quad p_g = \frac{|\mathcal{L}^K \cap \mathcal{I}_g|}{|\mathcal{I}_g|}

where :math:`p_g` is the ratio of recommended items from group :math:`g` to the total number of items in that group in the catalog.

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
