##############
Multiobjective
##############

**Multiobjective metrics** evaluate model performance across multiple, often conflicting, objectives simultaneously (e.g., Accuracy vs. Diversity). These metrics aggregate individual scores to assess the overall trade-off or distance from an ideal state.

Euclidean Distance
===============

Computes the **Euclidean Distance** between the model's performance and an Utopia Point (ideal performance). A lower value indicates the model is closer to the desired objectives.

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

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        complex_metrics:
            - name: Hypervolume
              params:
                  metric_names: [Precision, Recall, nDCG]
                  nadir_points: [0.0, 0.0, 0.0]
                  higher_is_better: [True, True, True]
