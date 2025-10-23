###########################
Iterative Recommender Guide
###########################

In this section, we will guide you through the process of implementing an iterative
recommendation model using WarpRec. As an example, we will use the **BPR** algorithm
(Bayesian Personalized Ranking), which is a widely used model for implicit feedback.

Iterative models in WarpRec differ from classical models in that they learn their
parameters through multiple epochs and batch updates. You will need to define
trainable parameters (e.g., embeddings), a forward method, and a training step.

Prerequisites
-------------

Before implementing a custom iterative model, ensure that you are familiar with the
following components of WarpRec:

- **IterativeRecommender**: The base class for iterative models, providing utilities
  for training loops, batching, and prediction.
- **Interactions** and **Sessions**: Objects representing user-item interactions and
  sequential session data.
- **Model Registry**: A centralized system for registering and retrieving models.
- **Loss Functions**: WarpRec includes commonly used losses like `BPRLoss`.

.. important::

    In this guide, we will implement the BPR model using IterativeRecommender. Unlike
    classical models, learning occurs during training steps and over multiple epochs.

Step 1: Define Your Model Class
-------------------------------

Start by creating a new Python class that inherits from **IterativeRecommender**.
You should:

1. Annotate all model parameters as class attributes.
2. Implement the `__init__` method to define learnable parameters and initialize
   embeddings.

.. warning::

    IterativeRecommenders expects certain parameters to be set, namely:
        - `weight_decay`: Weight decay for regularization.
        - `epochs`: Number of training epochs.
        - `learning_rate`: Learning rate for the optimizer.

    Not providing these parameters will result in unexpected behavior.

.. code-block:: python

    import torch
    from torch import nn, Tensor
    from warprec.recommenders.base_recommender import IterativeRecommender
    from warprec.recommenders.losses import BPRLoss

    class MyBPR(IterativeRecommender):
        embedding_size: int
        weight_decay: float
        batch_size: int
        epochs: int
        learning_rate: float

        def __init__(
            self,
            params: dict,
            *args: Any,
            device: str = "cpu",
            seed: int = 42,
            info: dict = None,
            **kwargs: Any,
        ):
            super().__init__(params, device=device, seed=seed, *args, **kwargs)
            users = info.get("users", None)
            if not users:
                raise ValueError("Users value must be provided to correctly initialize the model.")
            items = info.get("items", None)
            if not items:
                raise ValueError("Items value must be provided to correctly initialize the model.")

            self.user_embedding = nn.Embedding(users, self.embedding_size)
            self.item_embedding = nn.Embedding(items, self.embedding_size)
            self.apply(self._init_weights)  # See Step 2 for details
            self.loss = BPRLoss()
            self.to(self._device)

Step 2: Initialize Weights
--------------------------

WarpRec does not impose any specific weight initialization scheme. The common pattern used inside our implementaitons is to define a protected method `_init_weights` and apply it to the model using `self.apply(self._init_weights)`.

.. code-block:: python

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)

Step 3: Define the Training Step
--------------------------------

Iterative models rely on batches of samples to update the parameters. To ease the process, WarpRec defines three main methods that you need to implement:

1. `get_dataloader`: Returns a DataLoader that yields batches of training samples.

.. code-block:: python

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return interactions.get_pos_neg_dataloader(self.batch_size)

.. important::

    In this example we use a built-in method `get_pos_neg_dataloader` from the Interactions class.
    This method generates batches of (user, positive item, negative item) tuples for training for BPR or similar models.
    In some case you might want to implement your own dataloader.

2. `forward`: Defines the forward pass of the model. The input and output can differ based on the model.

.. code-block:: python

    def forward(self, user: Tensor, item: Tensor):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

3. `train_step`: Implements a single training step using a batch of data. This method computes the loss and returns it for backpropagation.

.. code-block:: python

    def train_step(self, batch: Any, *args, **kwargs):
        user, pos_item, neg_item = [x.to(self._device) for x in batch]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss = self.loss(pos_item_score, neg_item_score)

        return loss

.. warning::

    The `train_step` method must return a scalar loss tensor. WarpRec handles the
    backpropagation and optimizer step automatically. If the returned loss is not a Tensor, an error will be raised.

Step 4: Implement Prediction Methods
-----------------------------------

Recommendation models should implement `predict_full` and optionally `predict_sampled`.
In case of the full prediction, the method should return scores for all items for a batch of users.
In case of sampled prediction, the method should return scores only for a subset of candidate items.
These methods use the learned embeddings to compute scores:

.. code-block:: python

    @torch.no_grad()
    def predict_full(self, user_indices: Tensor, *args, **kwargs):
        user_e_all = self.user_embedding.weight
        item_e_all = self.item_embedding.weight
        u_embeddings_batch = user_e_all[user_indices]
        predictions = torch.matmul(u_embeddings_batch, item_e_all.transpose(0, 1))
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(self, user_indices: Tensor, item_indices: Tensor, *args, **kwargs):
        user_embeddings = self.user_embedding(user_indices)
        candidate_item_embeddings = self.item_embedding(item_indices.clamp(min=0))
        predictions = torch.einsum("bi,bji->bj", user_embeddings, candidate_item_embeddings)
        return predictions.to(self._device)


.. important::

    All the steps for model registration and usage with configurations are the same as in the Classical Recommender Guide.
