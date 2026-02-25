.. _evaluation:

##########
Evaluation
##########

.. toctree::
   :hidden:
   :maxdepth: 2

   metric
   metrics/accuracy
   metrics/rating
   metrics/coverage
   metrics/novelty
   metrics/diversity
   metrics/bias
   metrics/fairness
   metrics/multiobjective
   stat
   implement
   proxy_evaluation
   guide/sample
   guide/block

The WarpRec ``Evaluator`` is engineered for **high-throughput and efficiency** in metric computation, diverging significantly from the conventional methods employed by many existing frameworks.

Optimizing Metric Computation: Batch-Oriented Architecture
----------------------------------------------------------

Traditional frameworks often rely on a **dictionary-based approach** for representing ground-truth relevance, typically structured as follows:

.. code-block:: python

    {
      "user_id_1": [
        ("relevant_item_id_1", 1),
        ("relevant_item_id_2", 1),
      ],
      "user_id_2": [
        ("relevant_item_id_30", 1),
        ("relevant_item_id_15", 1),
      ],
      // ... more users
    }

While this structure is tenable for minimal datasets, its computational cost exhibits **poor scaling** as the number of users or the density of relevant items per user increases.

WarpRec addresses this limitation through a fundamental shift in its architectural design, leveraging two primary optimizations:

Tensor-Based Data Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WarpRec utilizes a **tensor-based data representation** in lieu of the dictionary structure. This vectorization of data is crucial for enabling **highly efficient retrieval of top-k items** and facilitating parallel operations, which are essential for performance at scale.

This approach is best illustrated by considering the batch-wise evaluation loop. Instead of relying on iterative lookups in a Python dictionary, all predictions and ground-truth data within a batch are represented as high-dimensional **PyTorch tensors**.

Example of Evaluation Flow (HitRate@k - Full Evaluation):
*********************************************************

Assume a scenario with a batch of size :math:`B=10` and a universe of :math:`N=10` items. Let the model's raw prediction scores and the binary ground-truth relevance be represented as tensors:

.. math::
    Pred =
    \begin{bmatrix}
    9.1 & 1.2 & 5.5 & 3.8 & 4.0 & 7.9 & 2.1 & 6.3 & 8.8 & 0.5 \\
    1.5 & 8.2 & 3.0 & 4.4 & 7.1 & 0.9 & 6.6 & 2.5 & 5.7 & 9.9 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
    \end{bmatrix} \in \mathbb{R}^{B \times N}

.. math::
    Target =
    \begin{bmatrix}
    1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
    \end{bmatrix} \in \{0,1\}^{B \times N}

For **HitRate@k** with :math:`k=3`, the evaluation proceeds as follows:

1. **Top-K Index Extraction:** Use tensor operations to retrieve indices of the top-:math:`k` predictions for each user:

.. math::
    \texttt{TOP\_K\_INDICES} =
    \begin{bmatrix}
    0 & 8 & 5 \\
    9 & 1 & 6 \\
    \vdots & \vdots & \vdots
    \end{bmatrix} \in \{0, \dots, N-1\}^{B \times k}

2. **Relevance Mapping:** Gather the corresponding binary relevance values:

.. math::
    \text{REL} =
    \begin{bmatrix}
    1 & 1 & 0 \\
    0 & 1 & 1 \\
    \vdots & \vdots & \vdots
    \end{bmatrix} \in \{0,1\}^{B \times k}

3. **Hit Calculation:** A user registers a hit if at least one of the top-:math:`k` items is relevant:

.. math::
    \texttt{HITS\_PER\_USER} = [\text{True}, \text{True}, \dots] \in \{0,1\}^B

4. **State Update:** Accumulate hits across the batch to update the metric's internal state.

The core of WarpRec's efficiency lies in its **batch-oriented approach**. A significant portion of recommender system metrics are evaluated **per-user**, as seen in the previous example. Processing the entire interaction or rating matrix simultaneously is often **computationally infeasible** due to size.

WarpRec segments the dataset into manageable **batches** for processing. This strategy dramatically enhances both **processing speed** and **memory efficiency** by localizing data access and computation.

Efficient Metric Aggregation: Single-Pass Computation
------------------------------------------------------

Traditional evaluation pipelines often suffer from inefficiency by evaluating metrics sequentially. This design necessitates **redundant iterations** over the user data, particularly when dealing with extensive user bases, leading to substantial overhead.

WarpRec mitigates this inefficiency by implementing a **single-pass metric computation** strategy. The system iterates through the batched data **only once**. During this iteration, it concurrently computes **partial results** for every configured metric.

These partial results are accumulated until the entire dataset has been processed, at which point the final, aggregated metric values are reported. This method significantly reduces the total execution time by eliminating repetitive data traversal.

As you can see in the example above, some intermediate values (e.g., TOP_K_INDICES, REL) are computed before evaluating the final value of a metric. These values are shared across the computation of different metrics, which means that evaluating multiple metrics will not slow down the overall process.

-----

Metrics Taxonomy
================

WarpRec includes **40 GPU-accelerated metrics** organized into 8 families.
All metrics are implemented as PyTorch modules and support distributed evaluation.

.. list-table::
   :header-rows: 1
   :widths: 16 18 50 16

   * - Family
     - Metric
     - Description
     - Type
   * - **Accuracy**
     - AUC
     - Area Under the ROC Curve.
     - Global
   * -
     - F1@K
     - Harmonic mean of Precision@K and Recall@K (or custom pair).
     - Top-K
   * -
     - GAUC
     - Per-user AUC averaged across all users.
     - Global
   * -
     - HitRate@K
     - Fraction of users with at least one relevant item in top K.
     - Top-K
   * -
     - LAUC@K
     - AUC limited to the top-K ranked items.
     - Top-K
   * -
     - MAP@K
     - Mean Average Precision rewarding higher-ranked correct items.
     - Top-K
   * -
     - MAR@K
     - Mean Average Recall indicating progressive retrieval quality.
     - Top-K
   * -
     - MRR@K
     - Mean Reciprocal Rank of the first relevant item.
     - Top-K
   * -
     - nDCG@K
     - Normalized Discounted Cumulative Gain (exponential relevance).
     - Top-K
   * -
     - nDCGRendle2020@K
     - nDCG with binary relevance following Rendle et al. (2020).
     - Top-K
   * -
     - Precision@K
     - Proportion of relevant items in the top K.
     - Top-K
   * -
     - Recall@K
     - Proportion of relevant items successfully retrieved in top K.
     - Top-K
   * - **Rating**
     - MAE
     - Mean Absolute Error between predicted and actual ratings.
     - Global
   * -
     - MSE
     - Mean Squared Error between predicted and actual ratings.
     - Global
   * -
     - RMSE
     - Root Mean Squared Error (same-scale error measure).
     - Global
   * - **Coverage**
     - ItemCoverage@K
     - Number of unique items recommended across all users.
     - Top-K
   * -
     - UserCoverage@K
     - Number of users with at least one recommended item.
     - Top-K
   * -
     - NumRetrieved@K
     - Average number of items retrieved per user in top K.
     - Top-K
   * -
     - UserCoverageAtN
     - Number of users with at least N items retrieved.
     - Top-K
   * - **Novelty**
     - EFD@K
     - Expected Free Discovery (log-discounted novelty).
     - Top-K
   * -
     - EPC@K
     - Expected Popularity Complement (linear novelty).
     - Top-K
   * - **Diversity**
     - Gini@K
     - Gini Index measuring inequality of item exposure.
     - Top-K
   * -
     - ShannonEntropy@K
     - Information entropy over item recommendation frequencies.
     - Top-K
   * -
     - SRecall@K
     - Subtopic Recall measuring feature/category coverage.
     - Top-K
   * - **Bias**
     - ACLT@K
     - Average Coverage of Long-Tail items in recommendations.
     - Top-K
   * -
     - APLT@K
     - Average Proportion of Long-Tail items per user.
     - Top-K
   * -
     - ARP@K
     - Average Recommendation Popularity of top-K items.
     - Top-K
   * -
     - PopREO@K
     - Popularity-based Ranking-based Equal Opportunity.
     - Top-K
   * -
     - PopRSP@K
     - Popularity-based Ranking-based Statistical Parity.
     - Top-K
   * - **Fairness**
     - BiasDisparityBD@K
     - Relative disparity between recommendation and training bias.
     - Top-K
   * -
     - BiasDisparityBR@K
     - Bias in recommendation frequency per user-item cluster pair.
     - Top-K
   * -
     - BiasDisparityBS
     - Bias in training data per user-item cluster pair.
     - Global
   * -
     - ItemMADRanking@K
     - Mean Absolute Deviation of discounted gain across item clusters.
     - Top-K
   * -
     - ItemMADRating@K
     - Mean Absolute Deviation of average ratings across item clusters.
     - Top-K
   * -
     - REO@K
     - Ranking-based Equal Opportunity across item clusters.
     - Top-K
   * -
     - RSP@K
     - Ranking-based Statistical Parity across item clusters.
     - Top-K
   * -
     - UserMADRanking@K
     - Mean Absolute Deviation of nDCG across user clusters.
     - Top-K
   * -
     - UserMADRating@K
     - Mean Absolute Deviation of average scores across user clusters.
     - Top-K
   * - **Multi-objective**
     - EucDistance@K
     - Euclidean distance from model performance to a Utopia Point.
     - Top-K
   * -
     - Hypervolume@K
     - Volume of objective space dominated relative to a Nadir Point.
     - Top-K
