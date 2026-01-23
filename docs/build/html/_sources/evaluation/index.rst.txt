.. _evaluation:

##########
Evaluation
##########

.. toctree::
   :hidden:
   :maxdepth: 2

   metric
   stat
   implement

The WarpRec ``Evaluator`` is engineered for **high-throughput and efficiency** in metric computation, diverging significantly from the conventional methods employed by many existing frameworks.

Optimizing Metric Computation: Batch-Oriented Architecture
----------------------------------------------------------

Traditional frameworks often rely on a **dictionary-based approach** for representing ground-truth relevance, typically structured as follows:

.. code-block:: json

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
    \text{TOP_K_INDICES} =
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
    \text{HITS_PER_USER} = [\text{True}, \text{True}, \dots] \in \{0,1\}^B

4. **State Update:** Accumulate hits across the batch to update the metric's internal state.

The core of WarpRec's efficiency lies in its **batch-oriented approach**. A significant portion of recommender system metrics are evaluated **per-user**, as seen in the previous example. Processing the entire interaction or rating matrix simultaneously is often **computationally infeasible** due to size.

WarpRec segments the dataset into manageable **batches** for processing. This strategy dramatically enhances both **processing speed** and **memory efficiency** by localizing data access and computation.

Efficient Metric Aggregation: Single-Pass Computation
------------------------------------------------------

Traditional evaluation pipelines often suffer from inefficiency by evaluating metrics sequentially. This design necessitates **redundant iterations** over the user data, particularly when dealing with extensive user bases, leading to substantial overhead.

WarpRec mitigates this inefficiency by implementing a **single-pass metric computation** strategy. The system iterates through the batched data **only once**. During this iteration, it concurrently computes **partial results** for every configured metric.

These partial results are accumulated until the entire dataset has been processed, at which point the final, aggregated metric values are reported. This method significantly reduces the total execution time by eliminating repetitive data traversal.

As you can see in the example above, some intermediate values (e.g., TOP\_K\_INDICES, REL) are computed before evaluating the final value of a metric. These values are shared across the computation of different metrics, which means that evaluating multiple metrics will not slow down the overall process.
