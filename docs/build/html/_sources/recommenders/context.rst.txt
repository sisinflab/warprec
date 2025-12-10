######################
Context-Aware Recommenders
######################

The **Context Recommenders** module of WarpRec contains models designed to incorporate contextual information (e.g., time, location, device, session data) into the recommendation process.
Unlike collaborative filtering recommenders that rely solely on User-Item interactions, these models leverage additional dimensions to improve prediction accuracy in specific situations.

In the following sections, you will find the list of available context-aware models within WarpRec, together with their respective parameters.

===================
Factorization-Based
===================

Factorization-Based context models extend standard matrix factorization techniques to handle multidimensional data (tensors) or feature vectors that include contextual variables.

- FM (Factorization Machines):
    A generic approach that mimics most factorization models by feature engineering. It models all nested interactions between variables using factorized parameters.
    In WarpRec, this implementation explicitly models the interactions between users, items, and available contextual features using second-order factorized interactions.
    It is particularly effective for sparse datasets with many categorical context features. **This model requires contextual information to function properly.**

.. code-block:: yaml

    models:
      FM:
        embedding_size: 64
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- NFM (Neural Factorization Machines):
    An extension of Factorization Machines that seamlessly combines the linearity of FM in modeling second-order feature interactions and the non-linearity of neural networks in modeling higher-order feature interactions.
    It replaces the standard second-order interaction term of FM with a "Bi-Interaction Pooling" layer followed by a Multi-Layer Perceptron (MLP).
    This allows the model to capture more complex and non-linear dependencies between users, items, and context features. **This model requires contextual information to function properly.**

.. code-block:: yaml

    models:
      NFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

===============================
Summary of Available Context-Aware Models
===============================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Model
     - Description
   * - Factorization-Based
     - FM
     - Factorization Machine modeling second-order interactions between user, item, and context.
   * - Factorization-Based
     - NFM
     - Neural Factorization Machine using MLP to model higher-order interactions between features.
