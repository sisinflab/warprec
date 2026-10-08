# Custom Recommender Implementation Guide

WarpRec provides a flexible and modular framework that allows you to implement your own recommendation models. This guide is designed to help you understand how to create custom recommenders and integrate them seamlessly into the WarpRec ecosystem.

The guide will focus on two main types of models:

1. **Classical Models**
   These are traditional recommendation algorithms that often rely on matrix operations, item similarity, or other non-iterative methods. They are typically fast to train and serve as a good starting point for exploring custom recommenders.

2. **Iterative Models**
   These models learn their parameters through iterative optimization, often involving multiple passes over the data and convergence criteria. Iterative models are more flexible and can capture complex patterns, but they may require more careful configuration and computational resources.

By exploring both classical and iterative approaches, you will gain a comprehensive understanding of how to implement custom recommenders in WarpRec, and how to leverage the framework's utilities for configuration management, device handling, and integration with advanced workflows such as hyperparameter optimization.

## General Usage

Besides the specific implementation details for each type of model, there are some general information that is useful to know when implementing a custom recommender.

### Registering Your Model

WarpRec uses a model registry to manage available models. Ensure that your class is decorated with `@model_registry.register(name="MyModel")`. This allows the framework to discover and use the model automatically.

```python
from warprec.utils.registry import model_registry

@model_registry.register(name="MyModel")
class MyModel(Recommender):
  # Rest of the model implementation...
```

### Add the Model to your Configuration

In case you are using configuration files to execute your experiment, you need an extra step to ensure that WarpRec recognizes your model. To use your custom model in a WarpRec pipeline, add it to your configuration file under the `general` section. You have two options:

- Create a script with only the model definition and pass it to the configuration.
- Create a fully fledged module (which can also include other models and metrics) and pass it to the configuration.

```yaml
general:
    custom_modules: [my_script.py, my_module]
```

!!! note

    WarpRec supports both relative and absolute paths for custom model scripts and modules. Also, you can pass a list of scripts/modules if needed. Organize your models as you prefer.

---

## Classic Recommender Guide

In this section, we will guide you through the process of implementing a classical recommendation model using WarpRec. As an example, we will use the **EASE** algorithm, which is a simple yet effective item-based model.

Classical models in WarpRec typically perform all learning operations during initialization, without iterative training loops.

### Prerequisites

Before implementing a custom model, ensure that you are familiar with the following components of WarpRec:

- **Recommender**: The base abstract class for all models, providing utilities for parameter management, device handling, and top-k recommendation generation.
- **ItemSimRecommender**: A specialized interface for models that learn an item similarity matrix.
- **Interactions**: The object representing user-item interactions in sparse format.
- **Model Registry**: A centralized system for registering and retrieving models.

!!! important

    In this guide, we will implement the EASE model using ItemSimRecommender interface. The base class will handle the prediction logic for us, in the Iterative Recommender Guide we will see how to implement custom prediction methods.

### Step 1: Define Your Model Class

Start by creating a new Python class that inherits from **ItemSimRecommender** (or **Recommender** if your model does not rely on item similarity). You should:

1. Annotate all model parameters as class attributes. For EASE, the only parameter is `l2`.
2. Implement the `__init__` method, which will initialize the model and perform the learning step directly.

```python
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.dataset import Interactions

class MyEASE(ItemSimRecommender):
    l2: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, interactions, *args, seed=seed, **kwargs)
```

### Step 2: Access Dataset Information

To access the sparse interaction matrix in CSR format, we are going to use the `Interactions` object passed in the constructor:

```python
X = interactions.get_sparse()
```

### Step 3: Compute the Item Similarity Matrix

EASE computes its item similarity matrix B directly using a closed-form solution:

1. Compute the Gram matrix of interactions: `G = X.T @ X + l2 * I`.
2. Invert the Gram matrix: `B = np.linalg.inv(G)`.
3. Normalize by the diagonal: `B /= -np.diag(B)`.
4. Set the diagonal to zero: `np.fill_diagonal(B, 0)`.

In code, this is how your implementation should look like:

```python
G = X.T @ X + self.l2 * np.identity(X.shape[1])
B = np.linalg.inv(G)
B /= -np.diag(B)
np.fill_diagonal(B, 0.0)

self.item_similarity = B
```

At this point, the model is fully initialized and ready to produce recommendations.

For reference, this is the complete implementation of the EASE model:

```python
@model_registry.register(name="MyEASE")
class MyEASE(ItemSimRecommender):
    l2: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, interactions, *args, seed=seed, **kwargs)

        X = self.train_matrix
        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B
```

---

## Iterative Recommender Guide

In this section, we will guide you through the process of implementing an iterative recommendation model using WarpRec. As an example, we will use the **BPR** algorithm (Bayesian Personalized Ranking), which is a widely used model for implicit feedback.

Iterative models in WarpRec differ from classical models in that they learn their parameters through multiple epochs and batch updates. You will need to define trainable parameters (e.g., embeddings), a forward method, and a training step.

### Prerequisites

Before implementing a custom iterative model, ensure that you are familiar with the following components of WarpRec:

- **IterativeRecommender**: The base class for iterative models, providing utilities for training loops, batching, and prediction.
- **Interactions** and **Sessions**: Objects representing user-item interactions and sequential session data.
- **Model Registry**: A centralized system for registering and retrieving models.
- **Loss Functions**: WarpRec includes commonly used losses like `BPRLoss`.

### Step 1: Define Your Model Class

Start by creating a new Python class that inherits from **IterativeRecommender**. You should:

1. Annotate all model parameters as class attributes.
2. Implement the `__init__` method to define learnable parameters and initialize embeddings.

!!! warning

    IterativeRecommenders expects certain parameters to be set, namely:

    - `batch_size`: The batch size for training.
    - `epochs`: Number of training epochs.
    - `learning_rate`: Learning rate for the optimizer.

    Not providing these parameters will result in unexpected behavior.

```python
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
```

!!! important

    The item_embedding uses `padding_idx=self.n_items` to handle potential padding in the item indices. Inside WarpRec, items are indexed from `0` to `num_items - 1`, with `num_items` reserved for padding. Also, the `_init_weights` method is a common pattern for initializing model weights, but you can choose any initialization scheme you prefer.

### Step 2: Define the Training Step

Iterative models rely on batches of samples to update the parameters. To ease the process, WarpRec defines three main methods that you need to implement:

**1. `get_dataloader`**: Returns a DataLoader that yields batches of training samples.

```python
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
```

!!! important

    In this example we use a built-in method `get_contrastive_dataloader` from the Interactions class. This method generates batches of (user, positive item, negative item) tuples for training for BPR or similar models. In some case you might want to implement your own dataloader.

**2. `forward`**: Defines the forward pass of the model. The input and output can differ based on the model.

```python
def forward(self, user: Tensor, item: Tensor):
    user_e = self.user_embedding(user)
    item_e = self.item_embedding(item)

    return torch.mul(user_e, item_e).sum(dim=1)
```

**3. `training_step`**: Implements a single training step using a batch of data. Iterative models are PyTorch Lightning modules, so this is Lightning's `training_step(batch, batch_idx)`: it computes the loss and returns it.

```python
def training_step(self, batch: Any, batch_idx: int):
    user, pos_item, neg_item = batch

    pos_item_score = self.forward(user, pos_item)
    neg_item_score = self.forward(user, neg_item)
    loss = self.bpr_loss(pos_item_score, neg_item_score)
    self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    return loss
```

!!! warning

    The `training_step` method must return a scalar loss tensor. WarpRec handles the backpropagation and optimizer step automatically. Logging it with `self.log` is optional; a logged value appears on Lightning's progress bar and in every epoch's report to Ray Tune, next to the validation metrics.

### Step 3: Implement Prediction Methods

Recommendation models must implement a prediction method to generate scores for user-item pairs. You must implement the `predict` method to define how predictions are made. Normal behavior is to compute a full prediction over the batch of users if item_indices is None, or compute predictions only for the provided item indices:

```python
def predict(self, user_indices: Tensor, *args, item_indices: Optional[Tensor] = None, **kwargs):
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
```

`item_indices` must default to `None`: under full evaluation the evaluator does not pass it at all. Under sampled evaluation it holds each user's candidates, and shorter candidate lists are padded with the index `n_items`. The padded embedding row above makes those columns score zero; a model without one must keep the index in range (for example with `item_indices.clamp(max=self.n_items - 1)`), since the evaluator discards the padded scores anyway.

!!! important

    All the steps for model registration and usage with configurations are the same as in the Classical Recommender Guide.

### Complete Implementation

For reference, this is the complete implementation of the BPR model:

```python
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

    def training_step(self, batch: Any, batch_idx: int):
        user, pos_item, neg_item = batch

        # Compute BPR loss
        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss = self.bpr_loss(pos_item_score, neg_item_score)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    def predict(self, user_indices: Tensor, *args, item_indices: Optional[Tensor] = None, **kwargs):
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
```


---

## Validating Hyperparameters

Only the attributes annotated on the model class itself are read from `params`. Without anything else, WarpRec accepts any value for them, and the training pipeline logs that the model "is registered in the model registry, but not in parameter registry". To validate them, and to let each one take a search space (see [Models configuration](../configuration/models.md)), register a parameter class **named exactly like the model**:

```python
from pydantic import field_validator

from warprec.utils.config.common import validate_greater_than_zero
from warprec.utils.config.model_configuration import FLOAT_FIELD, INT_FIELD, RecomModel
from warprec.utils.registry import params_registry


@params_registry.register("MyBPR")
class MyBPR(RecomModel):
    embedding_size: INT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size", "batch_size", "epochs", "learning_rate")
    @classmethod
    def check_positive(cls, v, info):
        return validate_greater_than_zero(cls, v, info.field_name)
```

WarpRec finds the class through its own name, so it cannot be called `MyBPRParams`; keep it in a module of its own to avoid clashing with the model class.

!!! tip

    [Guide 15 · Custom models and metrics](../guides/custom-models-and-metrics.md) runs all of this end to end: a closed-form and an iterative model, a parameter class and a custom metric, loaded through `custom_modules`, checked against the built-in models they reproduce.
