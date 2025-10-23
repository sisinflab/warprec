#########
Metrics
#########

.. toctree::
   :hidden:
   :maxdepth: 2

   metrics/accuracy
   metrics/bias
   metrics/coverage
   metrics/diversity
   metrics/fairness
   metrics/novelty
   metrics/rating
   metrics/implement

In this section, we provide an overview of the metrics implemented in WarpRec and explain how the framework handles evaluation in different scenarios.

Evaluation Modes
----------------

WarpRec supports multiple evaluation modes to provide flexibility in assessing recommender system performance:

1. **Full Evaluation:**
   In this mode, metrics are computed using all available items in the candidate set for each user.
   Full evaluation is suitable when the item catalog is small or when exhaustive evaluation is required for rigorous analysis.

2. **Sampled Evaluation:**
   To improve efficiency on large item sets, WarpRec allows **sampled evaluation**, where each user is evaluated against a subset of negative items.
   The number of negative samples per user is configurable, enabling trade-offs between computational cost and statistical fidelity.

.. note::
    WarpRec also provides tools for performing a wide range of statistical significance tests, with support for optional correction methods.

Example configuration:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50, ...]    # Evaluate metrics at top-10, top-20, and top-50
        metrics: [Precision, Recall, nDCG, ...] # Metrics to compute
        strategy: full | sampled    # Strategy to use during evaluation
        num_negatives: 100  # Use 100 negative samples per user (sampled evaluation)

This configuration ensures that metrics are computed efficiently while enabling detailed per-user analysis. For more details about the evaluation configuration, check the documentation :ref:`here <evaluation_configuration>`.
