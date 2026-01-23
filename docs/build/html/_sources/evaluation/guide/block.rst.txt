#############
Metric Blocks
#############

Let us introduce the concept of a ``Metric Block``. As explained in the :ref:`evaluation <evaluation>` section, WarpRec organizes evaluation by computing shared data structures—referred to as Metric Blocks—across different metrics.

This design does not increase the computational cost of evaluating individual metrics; instead, it enables the reuse of precomputed information, reducing redundancy and ensuring efficient data sharing throughout the entire evaluation process.

Using Metric Blocks
-------------------

To leverage a **Metric Block**, the implementation of a custom metric must be slightly refactored.
The first step is to declare the required blocks within the metric class:

.. code-block:: python

    from typing import Set
    from warprec.utils.enums import MetricBlock

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

These components correspond to the same tensors that were explicitly computed in the previous implementation.
However, when using Metric Blocks, WarpRec manages their computation and storage, and the metric only needs to retrieve them.

The initialization logic remains unchanged.
The only modification is in the ``.update()`` method, where the metric state is updated by consuming precomputed blocks rather than recalculating them.
An example implementation is shown below:

.. code-block:: python

    def update(self, preds: Tensor, **kwargs: Any):
        """Update the metric state with a new batch of predictions."""
        # 1. Retrieve precomputed blocks
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        # 2. Update internal accumulators
        hits += top_k_rel.sum(dim=1) / self.k # Precision contribution per user
        self.hits.index_add_(0, user_indices, batch_scores)
        users += self.valid_users(target)
        self.users.index_add_(0, user_indices, users)

With these adjustments, the metric now operates on shared data structures, eliminating redundant computations and improving evaluation efficiency.

Available Metric Blocks
-----------------------

WarpRec provides a set of **Metric Blocks**, i.e., reusable intermediate computations that can be shared across multiple metrics.
By precomputing these components once and making them available to all metrics, evaluation becomes more efficient and avoids redundant tensor operations.

The following table summarizes the available Metric Blocks:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Metric Block**
     - **Description**
   * - ``BINARY_RELEVANCE``
     - Relevance encoded as a binary tensor ``[0, 1]``, where 1 indicates that the item is relevant and 0 otherwise.
       Dim: ``[batch_size, num_items]``.
   * - ``DISCOUNTED_RELEVANCE``
     - Relevance values adjusted by a discounting factor (e.g., logarithmic), typically used in ranking metrics such as nDCG.
       Dim: ``[batch_size, num_items]``.
   * - ``VALID_USERS``
     - The number (or mask) of users that have at least one relevant item in the evaluation set.
       This block ensures that metrics are computed only on meaningful user subsets.
       Returns the number of valid users in the batch.
   * - ``TOP_K_INDICES``
     - The indices of the top-:math:`k` predictions returned by the model for each user.
       Dim: ``[batch_size, top_k]``.
   * - ``TOP_K_VALUES``
     - The actual prediction scores of the top-:math:`k` items for each user, aligned with ``TOP_K_INDICES``.
       Dim: ``[batch_size, top_k]``.
   * - ``TOP_K_BINARY_RELEVANCE``
     - The binary relevance (``[0, 1]``) of the top-:math:`k` predicted items, used in precision, recall, and hit-rate computations.
       Dim: ``[batch_size, top_k]``.
   * - ``TOP_K_DISCOUNTED_RELEVANCE``
     - The discounted relevance values of the top-:math:`k` predicted items, used in ranking-aware metrics such as nDCG.
       Dim: ``[batch_size, top_k]``.
