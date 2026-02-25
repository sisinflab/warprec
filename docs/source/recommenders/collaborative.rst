######################################
Collaborative-Filtering Recommenders
######################################

.. py:module:: warprec.recommenders.collaborative_filtering_recommender

.. autosummary::
   :nosignatures:

   autoencoder.ease.EASE
   autoencoder.elsa.ELSA
   autoencoder.cdae.CDAE
   autoencoder.macridvae.MacridVAE
   autoencoder.multidae.MultiDAE
   autoencoder.multivae.MultiVAE
   autoencoder.sansa.SANSA
   graph_based.dgcf.DGCF
   graph_based.egcf.EGCF
   graph_based.esigcf.ESIGCF
   graph_based.gcmc.GCMC
   graph_based.lightccf.LightCCF
   graph_based.lightgcl.LightGCL
   graph_based.lightgcn.LightGCN
   graph_based.lightgcnpp.LightGCNpp
   graph_based.lightgode.LightGODE
   graph_based.mixrec.MixRec
   graph_based.ngcf.NGCF
   graph_based.rp3beta.RP3Beta
   graph_based.sgcl.SGCL
   graph_based.sgl.SGL
   graph_based.ultragcn.UltraGCN
   graph_based.xsimgcl.XSimGCL
   knn.itemknn.ItemKNN
   knn.userknn.UserKNN
   latent_factor.admmslim.ADMMSlim
   latent_factor.bpr.BPR
   latent_factor.fism.FISM
   latent_factor.slim.Slim
   neural.convncf.ConvNCF
   neural.neumf.NeuMF

The **Collaborative-Filtering Recommenders** module of WarpRec is a collection of collaborative models.
In the following sections you will find the list of available models within WarpRec, together with their respective parameters.
These models can be used as-is or customized to fit experimental needs.

=============
Autoencoders
=============

.. py:module:: warprec.recommenders.collaborative_filtering_recommender.autoencoder

Autoencoder models learn compact latent representations of users or items by reconstructing user-item interaction data.
These models are particularly effective in sparse recommendation settings.

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.ease.EASE
   :members:
   :undoc-members:
   :show-inheritance:

- EASE (Embarrassingly Shallow Autoencoder):
  A simple, closed-form linear model that uses ridge regression to learn item-item similarities. Highly efficient and effective as a collaborative filtering baseline.

  For further details, please refer to the `paper <https://arxiv.org/abs/1905.03375>`_.

  .. code-block:: yaml

      models:
        EASE:
          l2: 10

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.elsa.ELSA
   :members:
   :undoc-members:
   :show-inheritance:

- ELSA (Efficient Linear Sparse Autoencoder):
  ELSA is a scalable approximation of the EASE algorithm that replaces the computationally expensive $O(I^3)$ matrix inversion with a sparse, low-rank decomposition optimized via stochastic gradient descent. This allows it to deliver EASE-level recommendation quality for massive item catalogs while significantly reducing memory usage and training time.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3523227.3551482>`_.

  .. code-block:: yaml

      models:
        ELSA:
          n_dims: 64
          batch_size: 2048
          epochs: 200
          learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.cdae.CDAE
   :members:
   :undoc-members:
   :show-inheritance:

- CDAE (Collaborative Denoising Auto-Encoder):
  A denoising autoencoder that specifically incorporates a user-specific latent vector (bias) into the hidden layer. This allows the model to capture user-specific patterns more effectively than standard autoencoders, making it highly effective for top-N recommendation tasks.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2835776.2835837>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.macridvae.MacridVAE
   :members:
   :undoc-members:
   :show-inheritance:

- MacridVAE (Macro-Disentangled Variational Autoencoder):
  A disentangled representation learning model that assumes user intentions are driven by a few macro concepts. It uses a VAE architecture with a specific encoder to separate these high-level concepts, improving interpretability and robustness.

  For further details, please refer to the `paper <https://arxiv.org/abs/1910.14238>`_.

  .. code-block:: yaml

    models:
      MacridVAE:
        embedding_size: 64
        encoder_hidden_dims: [600]
        k_fac: 7
        tau: 0.1
        corruption: 1.0
        nogb: False
        std: 0.075
        anneal_cap: 0.2
        total_anneal_steps: 200000
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.multidae.MultiDAE
   :members:
   :undoc-members:
   :show-inheritance:

- MultiDAE (Multinomial Denoising Autoencoder):
  A deep autoencoder trained with dropout for denoising input data. Learns robust latent representations from implicit feedback using a multinomial loss.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.multivae.MultiVAE
   :members:
   :undoc-members:
   :show-inheritance:

- MultiVAE (Multinomial Variational Autoencoder):
  A probabilistic variant of MultiDAE that models uncertainty in user preferences via variational inference. Useful for capturing diverse user behaviors and providing more personalized recommendations.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_.

  .. code-block:: yaml

      models:
        MultiVAE:
          intermediate_dim: 600
          latent_dim: 200
          corruption: 1.0
          anneal_cap: 0.2
          anneal_step: 200000
          weight_decay: 0.0001
          batch_size: 2048
          epochs: 200
          learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.autoencoder.sansa.SANSA
   :members:
   :undoc-members:
   :show-inheritance:

- SANSA (Scalable Approximate NonSymmetric Autoencoder):
  SANSA is a collaborative filtering algorithm designed to handle massive datasets by bypassing the memory bottlenecks of traditional linear models through sparse matrix approximations and an $LDL^T$ decomposition. It enables the training of high-performance autoencoders on a single machine, even with millions of items, by maintaining a compact, end-to-end sparse end-to-end architecture.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3604915.3608827>`_.

  .. code-block:: yaml

      models:
        SANSA:
          l2: 10
          target_density: 0.01

==============
Graph Based
==============

.. py:module:: warprec.recommenders.collaborative_filtering_recommender.graph_based

Graph-based recommenders exploit the structure of the user-item interaction graph to infer relationships and make recommendations.
These models capture high-order proximity and implicit associations through walks or neighborhood propagation.
They are well-suited for uncovering complex patterns in sparse datasets.

.. warning::

  Graph-based models require PyTorch Geometric (PyG) dependencies to be installed correctly. Check the :ref:`installation guide <install_guide>` for more information on how to install them.

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.dgcf.DGCF
   :members:
   :undoc-members:
   :show-inheritance:

- DGCF (Disentangled Graph Collaborative Filtering):
  A graph-based model that disentangles user and item embeddings into multiple latent intents (factors) using an iterative routing mechanism. It encourages independence between factors via a distance correlation loss.

  For further details, please refer to the `paper <https://arxiv.org/abs/2007.01764>`_.

  .. code-block:: yaml

    models:
      DGCF:
        embedding_size: 64
        n_factors: 4
        n_layers: 3
        n_iterations: 2
        cor_weight: 0.01
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.egcf.EGCF
   :members:
   :undoc-members:
   :show-inheritance:

- EGCF (Embedding-Less Graph Collaborative Filtering):
  A simplified graph model that removes user embeddings, learning only item embeddings to reduce complexity. It employs a joint loss combining BPR and contrastive learning (InfoNCE) to ensure alignment and uniformity without data augmentation. Supports 'parallel' and 'alternating' propagation modes.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3701230>`_.

  .. code-block:: yaml

    models:
      EGCF:
        embedding_size: 64
        n_layers: 3
        ssl_lambda: 0.1
        temperature: 0.1
        mode: alternating
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.esigcf.ESIGCF
   :members:
   :undoc-members:
   :show-inheritance:

- ESIGCF (Extremely Simplified but Intent-enhanced Graph Collaborative Filtering):
  A simplified graph model that removes explicit user embeddings and utilizes Joint Graph Convolution (JoGCN) with hybrid normalization. It integrates intent-aware contrastive learning to capture user intents without requiring data augmentation.

  For further details, please refer to the `paper <https://www.sciencedirect.com/science/article/abs/pii/S0952197625025266>`_.

  .. code-block:: yaml

    models:
      ESIGCF:
        embedding_size: 64
        n_layers: 3
        ssl_lambda: 0.1
        can_lambda: 0.1
        temperature: 0.1
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.gcmc.GCMC
   :members:
   :undoc-members:
   :show-inheritance:

- GCMC (Graph Convolutional Matrix Completion):
  A graph autoencoder designed for explicit feedback. It treats different rating values as distinct edge types in the user-item graph and learns embeddings using a graph convolutional encoder. A decoder then predicts rating probabilities. **This model requires explicit ratings to function properly**.

  For further details, please refer to the `paper <https://arxiv.org/abs/1706.02263>`_.

  .. code-block:: yaml

    models:
      GCMC:
        embedding_size: 64
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.lightccf.LightCCF
   :members:
   :undoc-members:
   :show-inheritance:

- LightCCF (Light Contrastive Collaborative Filtering):
  A contrastive learning model that introduces a Neighborhood Aggregation (NA) loss. It brings users closer to their interacted items while pushing them away from other positive pairs (users and items) in the batch. It can work with a standard MF encoder (n_layers=0) or a GCN encoder.

  For further details, please refer to the `paper <https://arxiv.org/abs/2504.10113>`_.

  .. code-block:: yaml

    models:
      LightCCF:
        embedding_size: 64
        n_layers: 0
        alpha: 0.1
        temperature: 0.2
        reg_weight: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcl.LightGCL
   :members:
   :undoc-members:
   :show-inheritance:

- LightGCL (Simple Yet Effective Graph Contrastive Learning):
  A graph contrastive learning model that uses Singular Value Decomposition (SVD) to construct a global contrastive view. It contrasts the local graph view (GCN) with the global SVD view to enhance representation learning and robustness against noise.

  For further details, please refer to the `paper <https://arxiv.org/abs/2302.08191>`_.

  .. code-block:: yaml

    models:
      LightGCL:
        embedding_size: 64
        n_layers: 2
        q: 5
        ssl_lambda: 0.1
        temperature: 0.2
        dropout: 0.1
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcn.LightGCN
   :members:
   :undoc-members:
   :show-inheritance:

- LightGCN:
  A simplified graph convolutional network designed for collaborative filtering. It eliminates feature transformations and nonlinear activations, focusing solely on neighborhood aggregation.

  For further details, please refer to the `paper <https://arxiv.org/abs/2002.02126>`_.

  .. code-block:: yaml

      models:
        LightGCN:
          embedding_size: 64
          n_layers: 3
          reg_weight: 0.001
          batch_size: 2048
          epochs: 200
          learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgcnpp.LightGCNpp
   :members:
   :undoc-members:
   :show-inheritance:

- LightGCN++:
  An enhanced version of LightGCN that introduces asymmetric normalization (controlled by alpha and beta) and a residual connection to the initial embeddings (controlled by gamma). This allows the model to better adapt to the specific structural properties of the dataset.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3640457.3688176>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.lightgode.LightGODE
   :members:
   :undoc-members:
   :show-inheritance:

- LightGODE (Light Post-Training Graph-ODE):
  A highly efficient model that trains embeddings without graph convolution using alignment and uniformity losses. It applies a continuous Graph-ODE solver only during inference to incorporate high-order connectivity.

  For further details, please refer to the `paper <https://arxiv.org/abs/2407.18910>`_.

  .. code-block:: yaml

    models:
      LightGODE:
        embedding_size: 64
        gamma: 2.0
        t: 1.0
        n_ode_steps: 2
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.mixrec.MixRec
   :members:
   :undoc-members:
   :show-inheritance:

- MixRec (Individual and Collective Mixing):
  A graph-based model that employs dual mixing strategies (Individual and Collective) to augment embeddings. It uses a dual-mixing contrastive learning objective to enhance consistency between positive pairs while leveraging mixed negatives.

  For further details, please refer to the `paper <https://doi.org/10.1145/3696410.3714565>`_.

  .. code-block:: yaml

    models:
      MixRec:
        embedding_size: 64
        n_layers: 3
        ssl_lambda: 1.1
        alpha: 0.1
        temperature: 0.2
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.ngcf.NGCF
   :members:
   :undoc-members:
   :show-inheritance:

- NGCF (Neural Graph-based Collaborative Filtering):
  A neural graph-based collaborative filtering model that explicitly captures high-order connectivity by propagating embeddings through the user-item interaction graph.

  For further details, please refer to the `paper <https://arxiv.org/abs/1905.08166>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.rp3beta.RP3Beta
   :members:
   :undoc-members:
   :show-inheritance:

- RP3Beta:
  A graph-based collaborative filtering model that performs a biased random walk of length 3 on the user-item bipartite graph.

  For further details, please refer to the `paper <https://www.zora.uzh.ch/id/eprint/131338/1/TiiS_2016.pdf>`_.

  .. code-block:: yaml

      models:
        RP3Beta:
          k: 10
          alpha: 0.1
          beta: 0.1
          normalize: True

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.sgcl.SGCL
   :members:
   :undoc-members:
   :show-inheritance:

- SGCL (Supervised Graph Contrastive Learning):
  A unified framework that merges the recommendation task and self-supervised learning into a single supervised contrastive loss. It simplifies the training pipeline by removing the need for negative sampling and data augmentation.

  For further details, please refer to the `paper <https://arxiv.org/abs/2507.13336>`_.

  .. code-block:: yaml

    models:
      SGCL:
        embedding_size: 64
        n_layers: 3
        temperature: 0.1
        reg_weight: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.sgl.SGL
   :members:
   :undoc-members:
   :show-inheritance:

- SGL (Self-supervised Graph Learning):
  A graph-based model that augments the user-item graph structure (via Node Dropout, Edge Dropout, or Random Walk) to create auxiliary views for contrastive learning, improving robustness and accuracy.

  For further details, please refer to the `paper <https://arxiv.org/abs/2010.10783>`_.

  .. code-block:: yaml

    models:
      SGL:
        embedding_size: 64
        n_layers: 3
        ssl_tau: 0.2
        ssl_reg: 0.1
        dropout: 0.1
        aug_type: ED
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.ultragcn.UltraGCN
   :members:
   :undoc-members:
   :show-inheritance:

- UltraGCN:
  A simplified GCN model that skips explicit message passing during training. It approximates infinite-layer graph convolutions using a constraint loss objective that models both user-item and item-item relationships, resulting in high efficiency and scalability.

  For further details, please refer to the `paper <https://arxiv.org/abs/2110.15114>`_.

  .. code-block:: yaml

    models:
      UltraGCN:
        embedding_size: 64
        w_lambda: 1.0
        w_gamma: 1.0
        w_neg: 1.0
        ii_k: 10
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.graph_based.xsimgcl.XSimGCL
   :members:
   :undoc-members:
   :show-inheritance:

- XSimGCL:
  A graph contrastive learning model that simplifies graph augmentations by adding uniform noise to embeddings. It achieves state-of-the-art performance by regulating the uniformity of the learned representation.

  For further details, please refer to the `paper <https://arxiv.org/abs/2209.02544>`_.

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
          batch_size: 2048
          epochs: 200
          learning_rate: 0.001

=========================
KNN (K Nearest Neighbor)
=========================

.. py:module:: warprec.recommenders.collaborative_filtering_recommender.knn

KNN-based models generate recommendations by identifying the most similar users or items based on interaction patterns or side information.

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.knn.itemknn.ItemKNN
   :members:
   :undoc-members:
   :show-inheritance:

- ItemKNN:
  A collaborative item-based KNN model that recommends items similar to those the user has already interacted with.

  For further details, please refer to the `paper <http://ieeexplore.ieee.org/document/1167344/>`_.

  .. code-block:: yaml

      models:
        ItemKNN:
          k: 10
          similarity: cosine

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.knn.userknn.UserKNN
   :members:
   :undoc-members:
   :show-inheritance:

- UserKNN:
  A collaborative user-based KNN model that recommends items liked by similar users.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_.

  .. code-block:: yaml

      models:
        UserKNN:
          k: 10
          similarity: cosine

================
Latent Factor
================

.. py:module:: warprec.recommenders.collaborative_filtering_recommender.latent_factor

Latent factor recommenders decompose the user–item interaction matrix into lower-dimensional representations.
These models capture hidden patterns in user preferences and item characteristics, allowing for effective personalization.
They include factorization-based approaches, pairwise ranking models, and sparse linear methods that emphasize interpretability and scalability.

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.latent_factor.admmslim.ADMMSlim
   :members:
   :undoc-members:
   :show-inheritance:

- ADMMSlim:
  An efficient implementation of SLIM using the ADMM optimization algorithm. It learns a sparse item-to-item similarity matrix for the top-N recommendation, balancing interpretability and performance.

  For further details, please refer to the `paper <https://doi.org/10.1145/3336191.3371774>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.latent_factor.bpr.BPR
   :members:
   :undoc-members:
   :show-inheritance:

- BPR:
  A pairwise ranking model that optimizes the ordering of items for each user. BPR is particularly effective for implicit feedback and is trained to maximize the margin between positive and negative item pairs.

  For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618>`_.

  .. code-block:: yaml

      models:
        BPR:
          embedding_size: 64
          reg_weight: 0.001
          batch_size: 2048
          epochs: 200
          learning_rate: 0.001

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.latent_factor.fism.FISM
   :members:
   :undoc-members:
   :show-inheritance:

- FISM:
  A recommendation algorithm that models item-to-item similarity by learning latent representations of items. Instead of explicitly learning user embeddings, FISM represents each user as the weighted average of the items they have interacted with, enabling efficient and accurate personalized recommendations.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2487575.2487589>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.latent_factor.slim.Slim
   :members:
   :undoc-members:
   :show-inheritance:

- Slim:
  A collaborative filtering model that learns a sparse item similarity matrix using L1 and L2 regularization. SLIM directly models the relationship between items, making it highly interpretable and effective for top-N recommendation.

  For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/6137254>`_.

  .. code-block:: yaml

      models:
        Slim:
          l1: 0.2
          alpha: 0.1

===========
Neural
===========

.. py:module:: warprec.recommenders.collaborative_filtering_recommender.neural

Neural recommenders leverage deep learning architectures to model complex, non-linear interactions between users and items.

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.neural.convncf.ConvNCF
   :members:
   :undoc-members:
   :show-inheritance:

- ConvNCF:
  Utilizes the outer product of user and item embeddings to construct a 2D interaction map, which is processed by Convolutional Neural Networks (CNNs) to capture complex and localized patterns in user-item interactions. ConvNCF enhances the expressive power of neural collaborative filtering by modeling structured relationships, making it well-suited for scenarios where fine-grained interaction modeling is critical.

  For further details, please refer to the `paper <https://arxiv.org/abs/1808.03912>`_.

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

.. autoclass:: warprec.recommenders.collaborative_filtering_recommender.neural.neumf.NeuMF
   :members:
   :undoc-members:
   :show-inheritance:

- NeuMF:
  Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) to capture both linear and non-linear user-item interactions. NeuMF is a highly expressive model that can adapt to various patterns in user behavior, making it suitable for both implicit and explicit feedback scenarios.

  For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_.

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

====================================
Summary of Available General Models
====================================

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
     - ELSA
     - Scalable EASE approximation using sparse low-rank decomposition via SGD.
   * -
     - CDAE
     - Denoising autoencoder with user-specific latent vectors.
   * -
     - MacridVAE
     - Disentangled VAE modeling macro concepts for user intentions.
   * -
     - MultiDAE
     - Denoising autoencoder optimized for implicit data.
   * -
     - MultiVAE
     - Variational autoencoder modeling uncertainty in preferences.
   * -
     - SANSA
     - Scalable autoencoder using sparse matrix approximations and LDLT decomposition.
   * - Graph Based
     - DGCF
     - Disentangles embeddings into latent factors using iterative routing.
   * -
     - EGCF
     - Embedding-less graph model using contrastive learning.
   * -
     - ESIGCF
     - Simplified JoGCN with intent-aware contrastive learning.
   * -
     - GCMC
     - Graph autoencoder for explicit feedback using multi-relational convolutions.
   * -
     - LightCCF
     - Contrastive model with Neighborhood Aggregation loss (supports MF/GCN).
   * -
     - LightGCL
     - Contrastive learning using SVD for global view augmentation.
   * -
     - LightGCN
     - Simplified Graph convolutional neural network.
   * -
     - LightGCN++
     - Improved LightGCN with asymmetric normalization and residual connections.
   * -
     - LightGODE
     - Training-free graph convolution using post-training ODE solver.
   * -
     - MixRec
     - Dual mixing data augmentation with contrastive learning.
   * -
     - NGCF
     - Complex Graph convolutional neural network.
   * -
     - RP3Beta
     - Random walk model with popularity penalization.
   * -
     - SGCL
     - Unified supervised contrastive learning without negative sampling.
   * -
     - SGL
     - Self-supervised learning with graph structure augmentation (ED, ND, RW).
   * -
     - UltraGCN
     - Efficient GCN approximation using constraint losses without message passing.
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
