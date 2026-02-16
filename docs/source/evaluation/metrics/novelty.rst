##########
Novelty
##########

**Novelty metrics** assess the extent to which a recommender system suggests items that are **new or unexpected** to the user, beyond what is already popular or frequently consumed. These metrics are important for fostering **exploration and serendipity**, as highly novel recommendations can lead to delightful discoveries.

EFD (Expected Free Discovery)
=============================

Estimates the likelihood that users discover relevant but less popular (**unexpected**) items in their top-K recommendations, promoting **serendipity**.

For further details, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/2043932.2043955>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [EFD]

**Extended-EFD** is also available, meaning you can compute the EFD score using a discounted relevance value, as follows:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: EFD
              params:
                  relevance: discounted

EPC (Expected Popularity Complement)
====================================

Measures the average **complement of item popularity** in the top-K recommendations, encouraging exposure to less popular content.

For further details, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/2043932.2043955>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [EPC]

**Extended-EPC** is also available, meaning you can compute the EPC score using a discounted relevance value, like such:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: EPC
              params:
                  relevance: discounted
