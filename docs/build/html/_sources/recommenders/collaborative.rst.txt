######################
General Recommenders
######################

The **General Recommenders** module of WarpRec is a collection of collaborative and content-based models.
In the following sections you will find the list of available models within WarpRec, together with their respective parameters.
These models can be used as-is or customized to fit experimental needs.

=============
Autoencoders
=============

Autoencoder models learn compact latent representations of users or items by reconstructing user-item interaction data.
These models are particularly effective in sparse recommendation settings.

- EASE (Embarrassingly Shallow Autoencoder):
  A simple, closed-form linear model that uses ridge regression to learn item-item similarities. Highly efficient and effective as a collaborative filtering baseline.

.. code-block:: yaml

    models:
      EASE:
        l2: 10

- CDAE (Collaborative Denoising Auto-Encoder):
  A denoising autoencoder that specifically incorporates a user-specific latent vector (bias) into the hidden layer. This allows the model to capture user-specific patterns more effectively than standard autoencoders, making it highly effective for top-N recommendation tasks.

.. code-block:: yaml

    models:
      CDAE:
        embedding_size: 64
        corruption: 1.0
        hid_activation: relu
        out_activation: sigmoid
        loss_type: BCE
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- MultiDAE (Multinomial Denoising Autoencoder):
  A deep autoencoder trained with dropout for denoising input data. Learns robust latent representations from implicit feedback using a multinomial loss.

.. code-block:: yaml

    models:
      MultiDAE:
        intermediate_dim: 600
        latent_dim: 200
        corruption: 1.0
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- MultiVAE (Multinomial Variational Autoencoder):
  A probabilistic variant of MultiDAE that models uncertainty in user preferences via variational inference. Useful for capturing diverse user behaviors and providing more personalized recommendations.

.. code-block:: yaml

    models:
      MultiVAE:
        intermediate_dim: 600
        latent_dim: 200
        corruption: 1.0
        anneal_cap: 0.2
        anneal_step: 200
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

==============
Graph Based
==============

Graph-based recommenders exploit the structure of the user-item interaction graph to infer relationships and make recommendations.
These models capture high-order proximity and implicit associations through walks or neighborhood propagation.
They are well-suited for uncovering complex patterns in sparse datasets.

.. warning::

  Graph-based models require PyTorch Geometric (PyG) dependencies to be installed correctly. Check the :ref:`installation guide <install_guide>` for more information on how to install them.

- GCMC (Graph Convolutional Matrix Completion):
  A graph autoencoder designed for explicit feedback. It treats different rating values as distinct edge types in the user-item graph and learns embeddings using a graph convolutional encoder. A decoder then predicts rating probabilities. **This model requires explicit ratings to function properly**.

.. code-block:: yaml

  models:
    GCMC:
      embedding_size: 64
      reg_weight: 0.001
      weight_decay: 0.0001
      batch_size: 2048
      epochs: 200
      learning_rate: 0.001

- LightGCN:
  A simplified graph convolutional network designed for collaborative filtering. It eliminates feature transformations and nonlinear activations, focusing solely on neighborhood aggregation.

.. code-block:: yaml

    models:
      LightGCN:
        embedding_size: 64
        n_layers: 3
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- LightGCN++:
  An enhanced version of LightGCN that introduces asymmetric normalization (controlled by alpha and beta) and a residual connection to the initial embeddings (controlled by gamma). This allows the model to better adapt to the specific structural properties of the dataset.

.. code-block:: yaml

  models:
    LightGCNpp:
      embedding_size: 64
      n_layers: 3
      alpha: 0.5
      beta: -0.1
      gamma: 0.2
      reg_weight: 0.001
      batch_size: 2048
      epochs: 200
      learning_rate: 0.001

- NGCF (Neural Graph-based Collaborative Filtering):
  A neural graph-based collaborative filtering model that explicitly captures high-order connectivity by propagating embeddings through the user-item interaction graph.

.. code-block:: yaml

    models:
      NGCF:
        embedding_size: 64
        weight_size: [64, 64]
        node_dropout: 0.1
        message_dropout: 0.1
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- RP3Beta:
  A graph-based collaborative filtering model that performs a biased random walk of length 3 on the user-item bipartite graph.

.. code-block:: yaml

    models:
      RP3Beta:
        k: 10
        alpha: 0.1
        beta: 0.1
        normalize: True

- XSimGCL:
  A graph contrastive learning model that simplifies graph augmentations by adding uniform noise to embeddings. It achieves state-of-the-art performance by regulating the uniformity of the learned representation.

.. code-block:: yaml

    models:
      XSimGCL:
        embedding_size: 64
        n_layers: 3
        lambda_: 0.2
        eps: 0.2
        temperature: 0.2
        layer_cl: 2
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

=========================
KNN (K Nearest Neighbor)
=========================

KNN-based models generate recommendations by identifying the most similar users or items based on interaction patterns or side information.

- ItemKNN:
  A collaborative item-based KNN model that recommends items similar to those the user has already interacted with.

.. code-block:: yaml

    models:
      ItemKNN:
        k: 10
        similarity: cosine

- UserKNN:
  A collaborative user-based KNN model that recommends items liked by similar users.

.. code-block:: yaml

    models:
      UserKNN:
        k: 10
        similarity: cosine

================
Latent Factor
================

Latent factor recommenders decompose the user–item interaction matrix into lower-dimensional representations.
These models capture hidden patterns in user preferences and item characteristics, allowing for effective personalization.
They include factorization-based approaches, pairwise ranking models, and sparse linear methods that emphasize interpretability and scalability.

- ADMMSlim:
  An efficient implementation of SLIM using the ADMM optimization algorithm. It learns a sparse item-to-item similarity matrix for the top-N recommendation, balancing interpretability and performance.

.. code-block:: yaml

    models:
      ADMMSlim:
        lambda_1: 0.1
        lambda_2: 0.1
        alpha: 0.2
        rho: 0.35
        it: 10
        positive_only: False
        center_columns: False

- BPR:
  A pairwise ranking model that optimizes the ordering of items for each user. BPR is particularly effective for implicit feedback and is trained to maximize the margin between positive and negative item pairs.

.. code-block:: yaml

    models:
      BPR:
        embedding_size: 64
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- FISM:
  A recommendation algorithm that models item-to-item similarity by learning latent representations of items. Instead of explicitly learning user embeddings, FISM represents each user as the weighted average of the items they have interacted with, enabling efficient and accurate personalized recommendations.

.. code-block:: yaml

    models:
      FISM:
        embedding_size: 64
        alpha: 0.1
        split_to: 5
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- Slim:
  A collaborative filtering model that learns a sparse item similarity matrix using L1 and L2 regularization. SLIM directly models the relationship between items, making it highly interpretable and effective for top-N recommendation.

.. code-block:: yaml

    models:
      Slim:
        l1: 0.2
        alpha: 0.1

===========
Neural
===========

Neural recommenders leverage deep learning architectures to model complex, non-linear interactions between users and items.

- ConvNCF:
  Utilizes the outer product of user and item embeddings to construct a 2D interaction map, which is processed by Convolutional Neural Networks (CNNs) to capture complex and localized patterns in user-item interactions. ConvNCF enhances the expressive power of neural collaborative filtering by modeling structured relationships, making it well-suited for scenarios where fine-grained interaction modeling is critical.

.. code-block:: yaml

    models:
      ConvNCF:
        embedding_size: 64
        cnn_channels: [32, 64]
        cnn_kernels: [2, 2]
        cnn_strides: [1, 1]
        dropout_prob: 0.1
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

- NeuMF:
  Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) to capture both linear and non-linear user-item interactions. NeuMF is a highly expressive model that can adapt to various patterns in user behavior, making it suitable for both implicit and explicit feedback scenarios.

.. code-block:: yaml

    models:
      NeuMF:
        mf_embedding_size: 64
        mlp_embedding_size: 64
        mlp_hidden_size: [64, 32]
        mf_train: True
        mlp_train: True
        dropout: 0.1
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1

===============================
Summary of Available General Models
===============================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Model
     - Description
   * - Autoencoders
     - EASE
     - Linear autoencoder using ridge regression for item similarity.
   * -
     - CDAE
     - Denoising autoencoder with user-specific latent vectors.
   * -
     - MultiDAE
     - Denoising autoencoder optimized for implicit data.
   * -
     - MultiVAE
     - Variational autoencoder modeling uncertainty in preferences.
   * - Graph Based
     - GCMC
     - Graph autoencoder for explicit feedback using multi-relational convolutions.
   * -
     - LightGCN
     - Simplified Graph convolutional neural network.
   * -
     - LightGCN++
     - Improved LightGCN with asymmetric normalization and residual connections.
   * -
     - NGCF
     - Complex Graph convolutional neural network.
   * -
     - RP3Beta
     - Random walk model with popularity penalization.
   * -
     - XSimGCL
     - Graph contrastive learning with noise perturbation.
   * - KNN
     - ItemKNN
     - Item-based collaborative KNN using similarity metrics.
   * -
     - UserKNN
     - User-based collaborative KNN using historical interactions.
   * - Latent Factor
     - ADMMSlim
     - Sparse item similarity model optimized via ADMM.
   * -
     - BPR
     - Pairwise ranking model for implicit feedback.
   * -
     - FISM
     - Efficient item similarity model using weighted average as user embeddings.
   * -
     - SLIM
     - Interpretable item similarity model with L1/L2 regularization.
   * - Neural
     - ConvNCF
     - Applies CNNs to user-item embeddings outer product to capture structured interaction patterns.
   * -
     - NeuMF
     - Hybrid neural model combining GMF and MLP layers.
