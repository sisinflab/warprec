##########
Diversity
##########

**Diversity metrics** evaluate the **variety of items** within a user's recommendations or across the recommendations for a set of users. These metrics are crucial for preventing "**filter bubbles**" and ensuring that users are exposed to a broad range of items, potentially increasing serendipity and user satisfaction.

GiniIndex
=========

Measures the **inequality** in the distribution of recommended items; lower values indicate more equitable item exposure.

For further details, please refer to this `book <https://link.springer.com/rwe/10.1007/978-1-4939-7131-2_110158>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [Gini]

Shannon Entropy
===============

Quantifies the **diversity** of recommended items using information entropy; higher values reflect greater item variety.

For further details, please refer to this `book <https://link.springer.com/referenceworkentry/10.1007/978-1-4939-7131-2_110158>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [ShannonEntropy]

SRecall (Subtopic Recall)
=========================

Measures how many **distinct subtopics or categories** are covered in the recommendations compared to the relevant ones, which reflects diversity across semantic dimensions.

**Note:** This metric requires the user to provide side information (e.g., item categories).

For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/2795403.2795405>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [SRecall]
