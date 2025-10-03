##########
Coverage
##########

**Coverage metrics** assess the extent to which a recommender system is able to recommend items from the entire catalog. They measure the **diversity** of the items recommended and the **proportion of the item space** that the system can effectively explore. High coverage suggests that the system can offer a wide range of recommendations beyond just the most popular items.

ItemCoverage@k
==============

Measures the **number of unique items** recommended in the top-k positions across all users, indicating **catalog coverage**.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ItemCoverage]

UserCoverage@k
==============

Calculates the **number of users** with at least one recommended item in their top-k recommendations, indicating reach and usefulness.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverage]

NumRetrieved@k
==============

Counts the **total number of distinct items** retrieved in the top-k recommendations across all users.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [NumRetrieved]

UserCoverageAtN
===============

Measures the number of users for whom the recommender retrieves at least **N** items, reflecting system responsiveness or minimum output capability.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [UserCoverageAtN]
