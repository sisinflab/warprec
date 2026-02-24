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
Step 1: Define Your Model Class
-------------------------------

Start by creating a new Python class that inherits from **IterativeRecommender**.
You should:

1. Annotate all model parameters as class attributes.
2. Implement the `__init__` method to define learnable parameters and initialize
   embeddings.

.. warning::

    IterativeRecommenders expects certain parameters to be set, namely:
        - `batch_size`: The batch size for training.
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
        batch_size: int
        epochs: int
        learning_rate: float

        def __init__(
            self,
            params: dict,
            info: dict,
            *args: Any,
            seed: int = 42,
            **kwargs: Any,
        ):
            super().__init__(params, info, *args, seed=seed, **kwargs)

            # Embeddings
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_embedding = nn.Embedding(
                self.n_items + 1, self.embedding_size, padding_idx=self.n_items
            )

            # Init embedding weights
            self.apply(self._init_weights)
            self.bpr_loss = BPRLoss()

.. important::

    The item_embedding uses `padding_idx=self.n_items` to handle potential padding in the item indices.
    Inside WarpRec, items are indexed from `0` to `num_items - 1`, with `num_items` reserved for padding.
    Also, the `_init_weights` method is a common pattern for initializing model weights, but you can choose any initialization scheme you prefer.

Step 2: Define the Training Step
--------------------------------

Iterative models rely on batches of samples to update the parameters. To ease the process, WarpRec defines three main methods that you need to implement:

1. `get_dataloader`: Returns a DataLoader that yields batches of training samples.

.. code-block:: python

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

.. important::

    In this example we use a built-in method `get_contrastive_dataloader` from the Interactions class.
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
        user, pos_item, neg_item = batch

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss = self.bpr_loss(pos_item_score, neg_item_score)

        return loss

.. warning::

    The `train_step` method must return a scalar loss tensor. WarpRec handles the
    backpropagation and optimizer step automatically. If the returned loss is not a Tensor, an error will be raised.

Step 3: Implement Prediction Methods
-----------------------------------

Recommendation models must implement a prediction method to generate scores for user-item pairs. You must implement the `predict` method to define how predictions are made.
Normal behavior is to compute a full prediction over the batch of users if item_indices is None, or compute predictions only for the provided item indices:

.. code-block:: python

    def predict(self, user_indices: Tensor, *args, item_indices: Optional[Tensor], **kwargs):
        user_embeddings = self.user_embedding(user_indices)
        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"
        predictions = torch.einsum(einsum_string, user_embeddings, item_embeddings)
        return predictions

.. important::

    All the steps for model registration and usage with configurations are the same as in the Classical Recommender Guide.

For reference, this is the complete implementation of the EASE model:

.. code-block:: python

    @model_registry.register(name="MyBPR")
    class MyBPR(IterativeRecommender):
        embedding_size: int
        batch_size: int
        epochs: int
        learning_rate: float

        def __init__(
            self,
            params: dict,
            info: dict,
            *args: Any,
            seed: int = 42,
            **kwargs: Any,
        ):
            super().__init__(params, info, *args, seed=seed, **kwargs)

            # Embeddings
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_embedding = nn.Embedding(
                self.n_items + 1, self.embedding_size, padding_idx=self.n_items
            )

            # Init embedding weights
            self.apply(self._init_weights)
            self.bpr_loss = BPRLoss()

        def get_dataloader(
            self,
            interactions: Interactions,
            sessions: Sessions,
            **kwargs: Any,
        ):
            return interactions.get_contrastive_dataloader(
                batch_size=self.batch_size,
                **kwargs,
            )

        def train_step(self, batch: Any, *args, **kwargs):
            user, pos_item, neg_item = batch

            # Compute BPR loss
            pos_item_score = self.forward(user, pos_item)
            neg_item_score = self.forward(user, neg_item)
            loss = self.bpr_loss(pos_item_score, neg_item_score)

            return loss

        def forward(self, user: Tensor, item: Tensor) -> Tensor:
            user_e = self.user_embedding(user)
            item_e = self.item_embedding(item)

            return torch.mul(user_e, item_e).sum(dim=1)

        def predict(self, user_indices: Tensor, *args, item_indices: Optional[Tensor], **kwargs):
            user_embeddings = self.user_embedding(user_indices)
            if item_indices is None:
                # Case 'full': prediction on all items
                item_embeddings = self.item_embedding.weight[:-1, :]
                einsum_string = "be,ie->bi"
            else:
                # Case 'sampled': prediction on a sampled set of items
                item_embeddings = self.item_embedding(item_indices)
                einsum_string = "be,bse->bs"
            predictions = torch.einsum(einsum_string, user_embeddings, item_embeddings)
            return predictions
