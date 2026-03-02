##########
Coverage
##########

.. py:module:: warprec.evaluation.metrics.coverage

Summary
~~~~~~~

.. autosummary::

    item_coverage.ItemCoverage
    user_coverage.UserCoverage
    user_coverage_at_n.UserCoverageAtN
    numretrieved.NumRetrieved

**Coverage metrics** assess the extent to which a recommender system is able to recommend items from the entire catalog. They measure the **diversity** of the items recommended and the **proportion of the item space** that the system can effectively explore. High coverage suggests that the system can offer a wide range of recommendations beyond just the most popular items.

ItemCoverage@k
==============

.. module:: warprec.evaluation.metrics.coverage.item_coverage
.. autoclass:: ItemCoverage
    :members:
    :undoc-members:
    :show-inheritance:

Measures the **number of unique items** recommended in the top-k positions across all users, indicating **catalog coverage**.

.. math::

    \text{ItemCoverage@}K = \left|\bigcup_{u \in \mathcal{U}} \mathcal{L}_u^K\right|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemCoverage]

UserCoverage@k
==============

.. module:: warprec.evaluation.metrics.coverage.user_coverage
.. autoclass:: UserCoverage
    :members:
    :undoc-members:
    :show-inheritance:

Calculates the **number of users** with at least one recommended item in their top-k recommendations, indicating reach and usefulness.

.. math::

    \text{UserCoverage@}K = |\{u \in \mathcal{U} : |\mathcal{L}_u^K| > 0\}|

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverage]

NumRetrieved@k
==============

.. module:: warprec.evaluation.metrics.coverage.numretrieved
.. autoclass:: NumRetrieved
    :members:
    :undoc-members:
    :show-inheritance:

Counts the **total number of distinct items** retrieved in the top-k recommendations across all users.

For further details, please refer to the `link <https://github.com/RankSys/RankSys/blob/master/RankSys-metrics/src/main/java/es/uam/eps/ir/ranksys/metrics/basic/NumRetrieved.java>`_

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [NumRetrieved]

UserCoverageAtN
===============

.. module:: warprec.evaluation.metrics.coverage.user_coverage_at_n
.. autoclass:: UserCoverageAtN
    :members:
    :undoc-members:
    :show-inheritance:

Measures the number of users for whom the recommender retrieves at least **N** items, reflecting system responsiveness or minimum output capability.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverageAtN]
