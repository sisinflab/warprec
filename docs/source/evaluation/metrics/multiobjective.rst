##############
Multiobjective
##############

**Multiobjective metrics** evaluate model performance across multiple, often conflicting, objectives simultaneously (e.g., Accuracy vs. Diversity). These metrics aggregate individual scores to assess the overall trade-off or distance from an ideal state.

Euclidean Distance
===============

Computes the **Euclidean Distance** between the model's performance and an Utopia Point (ideal performance). A lower value indicates the model is closer to the desired objectives.

For further details, please refer to this `link <https://www.sciencedirect.com/science/article/abs/pii/S0020025516318035?fr=RR-2&ref=pdf_download&rr=9d06d2a68bb6e889>`_ and this `paper <https://dl.acm.org/doi/10.1145/3583780.3615010>`_.

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: EucDistance
              params:
                  metric_names: [Precision, Recall, nDCG]
                  utopia_points: [1.0, 1.0, 1.0]

Hypervolume
======================================================

Measures the **volume of the objective space** dominated by the model's performance relative to a Nadir Point (worst-case reference). A higher volume indicates a better trade-off between metrics.

For further details, please refer to this `book <https://link.springer.com/chapter/10.1007/978-3-540-70928-2_64>`_ and this `paper <https://dl.acm.org/doi/10.1145/3583780.3615010>`_..

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: Hypervolume
              params:
                  metric_names: [Precision, Recall, nDCG]
                  nadir_points: [0.0, 0.0, 0.0]
                  higher_is_better: [True, True, True]
