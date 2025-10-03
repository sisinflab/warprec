###################
Per-User Evaluation
###################

WarpRec also supports **per-user evaluation**, i.e., the computation of metrics at the user level rather than in aggregate form.
In the standard setup, as described in the :ref:`evaluation guide <evaluation_guide>`, metrics are aggregated using **accumulators**, which collect contributions from all users into global statistics.

However, in some scenarios—such as when assessing **statistical significance** or when analyzing **user-level performance distributions**—it is necessary to compute metrics individually for each user.
WarpRec allows this with minimal modifications to the implementation.

Implementation Details
----------------------

The main adjustment is in the initialization of accumulators.
Instead of maintaining a single scalar tensor, the metric must store a vector of size ``[num_users]`` that records the contribution of each user.

In addition, for compatibility, the metric must declare that it supports per-user computation by setting the following class-level constant:

.. code-block:: python

    _CAN_COMPUTE_PER_USER: bool = True

The constructor then adapts the internal state depending on whether per-user evaluation is requested:

.. code-block:: python

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            # Store one entry per user to accumulate individual contributions
            self.add_state(
                "correct", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )
        else:
            # Standard global accumulator
            self.add_state(
                "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

Update Method
-------------

In the ``.update()`` method, the accumulation strategy depends on whether per-user evaluation is enabled:

.. code-block:: python

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Update the metric state with a new batch of predictions."""
        # Standard metric update logic (e.g., computing top-k relevance)

        if self.compute_per_user:
            # Accumulate contributions for each user
            self.correct.index_add_(
                0, user_indices, top_k_rel.sum(dim=1).float()
            )
        else:
            # Aggregate globally
            self.correct += top_k_rel.sum().float()

Compute Method
--------------

Finally, the ``.compute()`` method must return either the global metric value or the full per-user tensor:

.. code-block:: python

    def compute(self):
        """Compute the final metric value."""
        if self.compute_per_user:
            # Return tensor of size [num_users] with per-user values
            precision = self.correct / self.k
        else:
            # Return scalar global value
            precision = (
                self.correct / (self.users * self.k)
                if self.users > 0
                else torch.tensor(0.0)
            ).item()
        return {self.name: precision}

With this design, WarpRec seamlessly supports both **global** and **per-user** evaluation, enabling fine-grained analysis when needed while retaining efficiency in standard aggregate computations.
