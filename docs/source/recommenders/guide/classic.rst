#########################
Classic Recommender Guide
#########################

In this section, we will guide you through the process of implementing a classical recommendation model using WarpRec. As an example, we will use the **EASE** algorithm, which is a simple yet effective item-based model.

Classical models in WarpRec typically perform all learning operations during initialization, without iterative training loops.

Prerequisites
-------------

Before implementing a custom model, ensure that you are familiar with the following
components of WarpRec:

- **Recommender**: The base abstract class for all models, providing utilities for
  parameter management, device handling, and top-k recommendation generation.
- **ItemSimRecommender**: A specialized interface for models that learn an item
  similarity matrix.
- **Interactions**: The object representing user-item interactions in sparse format.
- **Model Registry**: A centralized system for registering and retrieving models.

.. important::

    In this guide, we will implement the EASE model using ItemSimRecommender interface.
    The base class will handle the prediction logic for us, in the Iterative Recommender Guide we will see how to implement custom prediction methods.

Step 1: Define Your Model Class
-------------------------------

Start by creating a new Python class that inherits from **ItemSimRecommender** (or
**Recommender** if your model does not rely on item similarity). You should:

1. Annotate all model parameters as class attributes. For EASE, the only parameter is
   `l2`.
2. Implement the `__init__` method, which will initialize the model and perform the
   learning step directly.

.. code-block:: python

    from warprec.recommenders.base_recommender import ItemSimRecommender
    from warprec.data.dataset import Interactions

    class MyEASE(ItemSimRecommender):
        l2: float

        def __init__(
            self,
            params: dict,
            interactions: Interactions,
            *args: Any,
            device: str = "cpu",
            seed: int = 42,
            info: dict = None,
            **kwargs: Any,
        ):
            super().__init__(
                params, interactions, device=device, seed=seed, info=info, *args, **kwargs
            )
            self._name = "MyEASE"

Step 2: Access Dataset Information
----------------------------------

WarpRec provides a dictionary named info containing key dataset metadata, such as:

- items: Number of unique items.
- users: Number of unique users.
- features: Number of unique features.

For item similarity models, items must be provided (inside WarpRec pipelines this is handled by the framework), as it determines the size of the similarity matrix.

To access the sparse interaction matrix in CSR format, we are gonna use the ``Interactions`` object passed in the constructor:

.. code-block:: python

    X = interactions.get_sparse()

Step 3: Compute the Item Similarity Matrix
------------------------------------------

EASE computes its item similarity matrix B directly using a closed-form solution:

1. Compute the Gram matrix of interactions: ``G = X.T @ X + l2 * I``.
2. Invert the Gram matrix: ``B = np.linalg.inv(G)``.
3. Normalize by the diagonal and zero out the diagonal: ``B /= -np.diag(B)``.
4. Set the diagonal to zero: ``np.fill_diagonal(B, 0)``.

In code, this is how your Implementation should look like:

.. code-block:: python

    G = X.T @ X + self.l2 * np.identity(X.shape[1])
    B = np.linalg.inv(G)
    B /= -np.diag(B)
    np.fill_diagonal(B, 0.0)

    self.item_similarity = B

At this point, the model is fully initialized and ready to produce recommendations.
