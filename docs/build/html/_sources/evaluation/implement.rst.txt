.. _evaluation_guide:

####################
Implementation Guide
####################

.. toctree::
   :hidden:
   :maxdepth: 2

   guide/block
   guide/sample
   guide/user

In this section, we'll walk you through how to implement your own evaluation metric inside WarpRec.

Metric Interfaces
-----------------

First, import the main metric interfaces:

.. code-block:: python

    from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric

The **BaseMetric** interface is sufficient for metrics that **do not depend on the cutoff (K)**. If your metric is ranking-based and requires a cutoff, use the **TopKMetric** interface.

We'll use the interfaces to implement a simple metric: **Precision@K**.

For a single user, Precision@K is the number of relevant items actually recommended divided by the cutoff K. The system-wide Precision@K is the sum of all individual Precision@K values divided by the number of users.

Defining Accumulators
---------------------

Before starting the implementation, we need to define the variables we need to *accumulate* across batch iterations.

- The number of **hits** (relevant items recommended) is not fixed and must be accumulated.
- The number of **users** involved in the calculation is also not fixed and must be accumulated.

.. note::
    The number of users is not always fixed. We often only consider users who have **at least one relevant item** in the ground truth for evaluation. This ensures the denominator reflects only *valid* evaluation cases.

The initialization
------------------

We define our metric inheriting from ``TopKMetric`` and use ``self.add_state()`` to define our accumulators. Torchmetrics handles the distributed synchronization and resetting of these states automatically.

.. code-block:: python

    import torch
    from typing import Any
    from torch import Tensor
    from warprec.evaluation.metrics.base_metric import TopKMetric

    class MyMetric(TopKMetric):
        # Type hints for our accumulated states
        hits: Tensor
        users: Tensor

        def __init__(
            self,
            k: int,
            num_users: int,
            *args: Any,
            compute_per_user: bool = False,
            dist_sync_on_step: bool = False,
            **kwargs: Any,
        ):
            # Calls the parent constructor to set self.k and dist_sync_on_step
            super().__init__(k, dist_sync_on_step)

            # Initialize accumulators. dist_reduce_fx="sum" handles accumulation across devices.
            self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

The ``.update()`` method
------------------------

The ``.update()`` method receives the model's raw prediction tensor, ``preds``, of size ``[batch_size, num_items]``. We need to use this to update our accumulators (``hits`` and ``users``).

We first retrieve the ground truth (labeled ``ground`` in ``kwargs``) and ensure it's in a binary relevance format using the built-in ``self.binary_relevance()``:

.. code-block:: python

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # 1. Get ground truth and ensure binary relevance
        target: Tensor = kwargs.get("ground", torch.zeros_like(preds))
        target = self.binary_relevance(target)

        # 2. Compute hits: a [batch_size, top_k] tensor of 1s (hits) and 0s (misses)
        top_k_rel = self.top_k_relevance(preds, target, self.k)

        # 3. Accumulate results
        # Sum the hits across all users and ranks
        self.hits += top_k_rel.sum()

        # Count only valid users (those with at least one relevant item)
        self.users += self.valid_users(target)

The ``.compute()`` method
-------------------------

The ``.compute()`` method is called to calculate the final metric value based on the accumulated states. WarpRec expects this method to return a **dictionary** where the key is the metric's name and the value is the computed result.

.. code-block:: python

    def compute(self):
        """Computes the final metric value."""
        # Precision@K = Total Hits / (Total Users * K)
        precision = (
            self.hits / (self.users * self.k)
            if self.users > 0
            else torch.tensor(0.0)
        )

        # Return the final value as a dictionary
        return {self.name: precision.item()}

And that's it! Your custom **Precision@K** metric is complete. You can now register it in the *metric\_registry* to make it available for configuration during training.

This is the *basic* concept of metrics inside the WarpRec framework, in the following sections we will guide you through additional layers of complexity that you can add to you metric to make it more robust.
