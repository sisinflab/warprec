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

- DeepFM (Deep Factorization Machines):
    A neural architecture that integrates a Factorization Machine component to model low-order feature interactions and a Deep Neural Network to capture high-order interactions. Both components share the same input embedding layer and operate in parallel to predict the final score. **This model requires contextual information to function properly.**

.. code-block:: yaml

    models:
      DeepFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- FM (Factorization Machines):
    A general predictor that models all nested interactions between input variables using factorized parameters. It explicitly captures second-order interactions between users, items, and contextual features, making it effective for sparse datasets with categorical variables. **This model requires contextual information to function properly.**

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
    An extension of Factorization Machines that replaces the standard second-order interaction term with a "Bi-Interaction Pooling" layer followed by a Multi-Layer Perceptron (MLP). This architecture allows the model to capture complex, non-linear, and higher-order dependencies between features. **This model requires contextual information to function properly.**

.. code-block:: yaml

    models:
      NFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- WideDeep (Wide & Deep Learning):
    A framework that jointly trains a wide linear model for memorization and a deep neural network for generalization. It combines a generalized linear model (Wide) with a feed-forward neural network (Deep) to capture both low-order and high-order feature interactions. **This model requires contextual information to function properly.**

.. code-block:: yaml

    models:
      WideDeep:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
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
     - DeepFM
     - Parallel combination of FM and DNN to capture both low- and high-order feature interactions.
   * -
     - FM
     - Factorization Machine modeling second-order interactions between user, item, and context.
   * -
     - NFM
     - Neural Factorization Machine using MLP to model higher-order interactions between features.
   * -
     - WideDeep
     - Joint training of a linear model (Wide) and a Deep Neural Network (Deep) for memorization and generalization.
