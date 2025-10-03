##################
Sampled Evaluation
##################

WarpRec also supports **sampled evaluation** for metrics. In this approach, instead of evaluating the model performance across the entire set of items, the framework samples a fixed number of negative items for each user.

Thus, for each user, the evaluation is performed over a reduced set of items composed of the true positives and the sampled negatives, i.e. :math:`(positives + negatives)`.

However, this introduces a challenge: what happens if two or more users within the same batch have a different number of positive items?

To illustrate this issue, let us start with a toy example. Consider a dataset with 10 items (for simplicity of visualization) and a batch size of 5. The prediction tensor is shown below:

.. math::
   Pred_{\text{full}} = \begin{bmatrix}
      0.95 & 0.12 & 0.44 & 0.77 & 0.05 & 0.81 & 0.50 & -\infty & 0.69 & 0.21 \\
      0.10 & 0.88 & -\infty & 0.29 & 0.73 & -\infty & 0.91 & 0.03 & 0.47 & 0.62 \\
      0.58 & 0.25 & 0.99 & 0.14 & 0.83 & 0.37 & 0.60 & 0.07 & -\infty & 0.40 \\
      0.71 & 0.01 & 0.35 & 0.90 & -\infty & 0.52 & 0.20 & 0.85 & 0.49 & 0.66 \\
      0.18 & 0.65 & 0.30 & 0.87 & 0.54 & 0.09 & 0.78 & -\infty & 0.23 & 0.93
   \end{bmatrix}

.. math::
    Target_{\text{full}} = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}

In the case of full evaluation, interactions already observed during training are masked in the prediction tensor using :math:`-\infty`. This ensures that evaluation only considers the ranking of unseen items.

Suppose now we perform sampled evaluation with 2 negative samples per user. The resulting prediction tensor is significantly smaller:

.. math::
    Pred_{\text{sampled}} = \begin{bmatrix}
        0.95 & -\infty & 0.12 & 0.44 \\
        0.88 & 0.91 & 0.29 & 0.47 \\
        0.99 & 0.83 & 0.14 & 0.25 \\
        0.90 & -\infty & 0.20 & 0.49 \\
        0.87 & -\infty & 0.23 & 0.18
    \end{bmatrix}

.. math::
    Target_{\text{sampled}} = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        1 & 1 & 0 & 0 \\
        1 & 1 & 0 & 0 \\
        1 & 0 & 0 & 0 \\
        1 & 0 & 0 & 0
    \end{bmatrix}

In this setting, :math:`-\infty` is used for **padding**. Since WarpRec requires rectangular tensors, the number of positive labels must be padded to match across users.

The final evaluation results will differ from those obtained with full evaluation, but the main advantages of sampled evaluation are **reduced computational cost and lower memory usage**.

Implementation details
----------------------

When implementing a new metric, it is important to account for the possibility of sampled evaluation.

For instance, in the case of ``Precision@k`` (as implemented in this guide), no additional handling is required. This is because ``Precision@k`` evaluates *whether* a relevant item appears in the top-k predictions, without considering *which* specific items are recommended.

Conversely, if a metric depends on the actual indices of recommended items, the implementation must be adapted. An example modification is shown below:

.. code-block:: python

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Standard metric update code here

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

With this adjustment, the metric can correctly handle both full and sampled evaluation.
