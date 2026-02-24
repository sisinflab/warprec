.. _evaluation_guide:

####################
Implementation Guide
####################

.. toctree::
   :hidden:
   :maxdepth: 2

   guide/block
   guide/sample

In this section, we'll walk you through how to implement your own evaluation metric inside WarpRec.

Metric Interfaces
-----------------

First, import the main metric interfaces:

.. code-block:: python

    from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric, TopKMetric, RatingMetric, BaseMetric

The **BaseMetric** interface is sufficient for metrics that **do not depend on the cutoff (K)**.

If your metric is an error metric (e.g., MSE, MAE) that relies on raw ratings rather than ranking, use the **RatingMetric** interface.

If your metric is ranking-based and requires a cutoff, use the **TopKMetric** interface.

Moreover, if your metric needs to be computed on a per-user basis before aggregating to a system-wide value (e.g., average over users), use the **UserAverageTopKMetric** interface.

We'll use the **UserAverageTopKMetric** interface to implement a simple metric: **Precision@K**.

For a single user, Precision@K is the number of relevant items actually recommended divided by the cutoff K. The system-wide Precision@K is the sum of all individual Precision@K values divided by the number of users.

The implementation
---------------------

Before starting the implementation, we need to define the variables we need to *accumulate* across batch iterations.

For **Precision@K**, we need to accumulate just one value: the number of **hits** (relevant items recommended) per user. This value will be stored in a tensor of length **num_users** where each index corresponds to the number of hits for that given user.

Using the **UserAverageTopKMetric** this step is *extremely* simplified, we only need to implement the logic to retrieve the number of hits from the  top-k relevant item tensor, which can be implemented like this:

.. code-block:: python

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        return top_k_rel.sum(dim=1).float() / self.k

In this implementation, `top_k_rel` is a binary tensor of shape [batch_size, k] where each entry is 1 if the recommended item at that position is relevant for the user and 0 otherwise. By summing over the second dimension (k), we get the total number of hits for each user in the batch. Finally, we divide by `self.k` to get the Precision@K for each user.

For the **UserAverageTopKMetric** interface, we don't need to worry about accumulating the hits across batches or computing the final average. The framework will automatically handle the accumulation of scores and the final averaging over users at the end of the evaluation process.

.. note::
    The number of users is not always fixed. We often only consider users who have **at least one relevant item** in the ground truth for evaluation. This ensures the denominator reflects only *valid* evaluation cases.
