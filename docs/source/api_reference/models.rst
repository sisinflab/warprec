.. _api_models:

##################
Models Reference
##################

WarpRec ships with 57 built-in algorithms spanning 6 model families.
This page provides class signatures, loss functions, and hyperparameter tables for every model.

For quick-reference taxonomy tables, see :ref:`Recommenders <recommender>`.
For per-family documentation with usage examples, see :doc:`/recommenders/index`.
For configuration syntax, see :ref:`Configuration <configuration>`.

.. autoclass:: warprec.recommenders.base_recommender.Recommender
    :members:
    :show-inheritance:

.. autoclass:: warprec.recommenders.base_recommender.IterativeRecommender
    :members:
    :show-inheritance:

.. autoclass:: warprec.recommenders.base_recommender.ItemSimRecommender
    :members:
    :show-inheritance:

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Class Hierarchy
===============

All recommendation models inherit from one of the following base classes:

- **Recommender** — Base class for non-iterative models (e.g., KNN, popularity-based).
- **IterativeRecommender(Recommender)** — Base class for models trained via gradient descent with epochs, batch size, and learning rate.
- **ItemSimRecommender(Recommender)** — Base class for models that compute an item-item similarity matrix (e.g., EASE, Slim, RP3Beta).

Additionally, specialized mixins provide utility methods:

- **GraphRecommenderUtils** — Bipartite graph construction, sparse adjacency normalization, and PyG utilities for graph-based models.
- **SequentialRecommenderUtils** — Sequence padding, session extraction, and sliding window support for sequential models.
- **ContextRecommenderUtils** — Feature embedding tables and context-aware input processing.

-----

Loss Functions
==============

Models in WarpRec use the following loss functions:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Loss
     - Description
   * - **BPRLoss**
     - Bayesian Personalized Ranking loss. Maximizes the margin between positive and negative item scores: :math:`\mathcal{L}_\text{BPR} = -\sum \log \sigma(\hat{r}_{ui} - \hat{r}_{uj})`.
   * - **BCEWithLogitsLoss**
     - Binary Cross-Entropy with logits for pointwise classification.
   * - **MSELoss**
     - Mean Squared Error for rating prediction tasks.
   * - **EmbLoss**
     - L2 regularization on embedding parameters: :math:`\mathcal{L}_\text{emb} = \lambda \sum \|\mathbf{e}\|^2`.
   * - **InfoNCELoss**
     - Contrastive loss using temperature-scaled cosine similarity for self-supervised graph models.
   * - **ELBO**
     - Evidence Lower Bound for variational autoencoders: :math:`\mathcal{L}_\text{ELBO} = \mathbb{E}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))`.
   * - **MultiDAELoss / MultiVAELoss**
     - Custom reconstruction losses for multinomial denoising/variational autoencoders.
   * - **gBCELoss**
     - Group-wise Binary Cross-Entropy for gSASRec.
   * - **Alignment + Uniformity**
     - Combines alignment of positive pairs with uniformity of embedding distribution (LightGODE).
   * - **Supervised Contrastive**
     - Merges recommendation and self-supervised objectives into a single loss (SGCL).

-----

Unpersonalized Models (3)
=========================

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Model
     - Base Class
     - Hyperparameters
   * - **Pop**
     - Recommender
     - *(none — derived from data)*
   * - **Random**
     - Recommender
     - *(none)*
   * - **ProxyRecommender**
     - Recommender
     - ``recommendation_file`` (str), ``sep`` (str), ``header`` (bool)

-----

Content-Based Models (1)
========================

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Model
     - Base Class
     - Hyperparameters
   * - **VSM**
     - Recommender
     - ``similarity`` (str), ``user_profile`` (str), ``item_profile`` (str)

-----

Collaborative Filtering Models (31)
====================================

Autoencoders (7)
----------------

.. list-table::
   :header-rows: 1
   :widths: 12 12 12 64

   * - Model
     - Base Class
     - Loss
     - Hyperparameters
   * - **EASE**
     - ItemSimRecommender
     - Closed-form
     - ``l2``
   * - **ELSA**
     - IterativeRecommender
     - MSE
     - ``n_dims``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **CDAE**
     - IterativeRecommender
     - MSE/BCE
     - ``embedding_size``, ``corruption``, ``hid_activation``, ``out_activation``, ``loss_type``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **MacridVAE**
     - IterativeRecommender
     - ELBO + KL
     - ``embedding_size``, ``encoder_hidden_dims``, ``k_fac``, ``tau``, ``corruption``, ``nogb``, ``std``, ``anneal_cap``, ``total_anneal_steps``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **MultiDAE**
     - IterativeRecommender
     - MultiDAELoss
     - ``intermediate_dim``, ``latent_dim``, ``corruption``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **MultiVAE**
     - IterativeRecommender
     - MultiVAELoss
     - ``intermediate_dim``, ``latent_dim``, ``corruption``, ``anneal_cap``, ``anneal_step``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **SANSA**
     - ItemSimRecommender
     - Closed-form
     - ``l2``, ``target_density``

Graph-Based (16)
----------------

.. list-table::
   :header-rows: 1
   :widths: 12 12 76

   * - Model
     - Loss
     - Key Hyperparameters
   * - **DGCF**
     - BPR + Corr
     - ``embedding_size``, ``n_factors``, ``n_layers``, ``n_iterations``, ``cor_weight``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **EGCF**
     - BPR + InfoNCE
     - ``embedding_size``, ``n_layers``, ``ssl_lambda``, ``temperature``, ``mode``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **ESIGCF**
     - BPR + InfoNCE + CAN
     - ``embedding_size``, ``n_layers``, ``ssl_lambda``, ``can_lambda``, ``temperature``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **GCMC**
     - Softmax CE
     - ``embedding_size``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **LightCCF**
     - InfoNCE + NA
     - ``embedding_size``, ``n_layers``, ``alpha``, ``temperature``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **LightGCL**
     - BPR + InfoNCE
     - ``embedding_size``, ``n_layers``, ``q``, ``ssl_lambda``, ``temperature``, ``dropout``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **LightGCN**
     - BPR + Emb
     - ``embedding_size``, ``n_layers``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **LightGCN++**
     - BPR + Emb
     - ``embedding_size``, ``n_layers``, ``alpha``, ``beta``, ``gamma``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **LightGODE**
     - Align + Uniform
     - ``embedding_size``, ``gamma``, ``t``, ``n_ode_steps``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **MixRec**
     - BPR + InfoNCE
     - ``embedding_size``, ``n_layers``, ``ssl_lambda``, ``alpha``, ``temperature``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **NGCF**
     - BPR + Emb
     - ``embedding_size``, ``weight_size``, ``node_dropout``, ``message_dropout``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **RP3Beta**
     - Graph walk
     - ``k``, ``alpha``, ``beta``, ``normalize``
   * - **SGCL**
     - Supervised CL
     - ``embedding_size``, ``n_layers``, ``temperature``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **SGL**
     - BPR + InfoNCE
     - ``embedding_size``, ``n_layers``, ``ssl_tau``, ``ssl_reg``, ``dropout``, ``aug_type``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **UltraGCN**
     - Constraint loss
     - ``embedding_size``, ``w_lambda``, ``w_gamma``, ``w_neg``, ``ii_k``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **XSimGCL**
     - BPR + InfoNCE
     - ``embedding_size``, ``n_layers``, ``lambda_``, ``eps``, ``temperature``, ``layer_cl``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``

.. warning::

    Graph-based models require **PyTorch Geometric (PyG)** to be installed. See the :ref:`installation guide <install_guide>`.

KNN (2)
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Model
     - Hyperparameters
   * - **ItemKNN**
     - ``k`` (int), ``similarity`` (str: cosine, jaccard, etc.)
   * - **UserKNN**
     - ``k`` (int), ``similarity`` (str)

Latent Factor (4)
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Model
     - Loss
     - Hyperparameters
   * - **ADMMSlim**
     - ADMM
     - ``lambda_1``, ``lambda_2``, ``alpha``, ``rho``, ``it``, ``positive_only``, ``center_columns``
   * - **BPR**
     - BPR + Emb
     - ``embedding_size``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **FISM**
     - BCE + Emb
     - ``embedding_size``, ``alpha``, ``split_to``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **Slim**
     - ElasticNet
     - ``l1``, ``alpha``

Neural (2)
----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Model
     - Loss
     - Hyperparameters
   * - **ConvNCF**
     - BPR + Emb
     - ``embedding_size``, ``cnn_channels``, ``cnn_kernels``, ``cnn_strides``, ``dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``
   * - **NeuMF**
     - BCE + Emb
     - ``mf_embedding_size``, ``mlp_embedding_size``, ``mlp_hidden_size``, ``mf_train``, ``mlp_train``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``

-----

Context-Aware Models (8)
========================

All context-aware models require **contextual information** (side features) and assume a **Leave-One-Out** splitting strategy.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Model
     - Key Hyperparameters
   * - **AFM**
     - ``embedding_size``, ``attention_size``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **DCN**
     - ``embedding_size``, ``mlp_hidden_size``, ``cross_layer_num``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **DCNv2**
     - ``embedding_size``, ``mlp_hidden_size``, ``cross_layer_num``, ``dropout``, ``model_structure``, ``use_mixed``, ``expert_num``, ``low_rank``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **DeepFM**
     - ``embedding_size``, ``mlp_hidden_size``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **FM**
     - ``embedding_size``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **NFM**
     - ``embedding_size``, ``mlp_hidden_size``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **WideAndDeep**
     - ``embedding_size``, ``mlp_hidden_size``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``
   * - **xDeepFM**
     - ``embedding_size``, ``mlp_hidden_size``, ``cin_layer_size``, ``direct``, ``dropout``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``

All context-aware models use **BCEWithLogitsLoss + EmbLoss**.

-----

Sequential Models (10)
=======================

All sequential models inherit from ``SequentialRecommenderUtils`` and share the ``max_seq_len`` and ``neg_samples`` parameters.

.. list-table::
   :header-rows: 1
   :widths: 14 14 72

   * - Model
     - Architecture
     - Key Hyperparameters
   * - **Caser**
     - CNN
     - ``embedding_size``, ``n_h``, ``n_v``, ``dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **FOSSIL**
     - Markov Chain
     - ``embedding_size``, ``order_len``, ``alpha``, ``reg_weight``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **GRU4Rec**
     - RNN (GRU)
     - ``embedding_size``, ``hidden_size``, ``num_layers``, ``dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **NARM**
     - RNN + Attention
     - ``embedding_size``, ``hidden_size``, ``n_layers``, ``hidden_dropout_prob``, ``attn_dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **BERT4Rec**
     - Transformer (bidir.)
     - ``embedding_size``, ``n_layers``, ``n_heads``, ``inner_size``, ``dropout_prob``, ``attn_dropout_prob``, ``mask_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **CORE**
     - Transformer (RCE)
     - ``embedding_size``, ``dnn_type``, ``n_layers``, ``n_heads``, ``inner_size``, ``hidden_dropout_prob``, ``attn_dropout_prob``, ``session_dropout``, ``item_dropout``, ``temperature``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **gSASRec**
     - Transformer (gBCE)
     - ``embedding_size``, ``n_layers``, ``n_heads``, ``inner_size``, ``dropout_prob``, ``attn_dropout_prob``, ``gbce_t``, ``reuse_item_embeddings``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **LightSANs**
     - Transformer (low-rank)
     - ``embedding_size``, ``n_layers``, ``n_heads``, ``k_interests``, ``inner_size``, ``dropout_prob``, ``attn_dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **LinRec**
     - Linear Attention
     - ``embedding_size``, ``n_heads``, ``inner_size``, ``dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``
   * - **SASRec**
     - Transformer (causal)
     - ``embedding_size``, ``n_layers``, ``n_heads``, ``inner_size``, ``dropout_prob``, ``attn_dropout_prob``, ``reg_weight``, ``weight_decay``, ``batch_size``, ``epochs``, ``learning_rate``, ``neg_samples``, ``max_seq_len``

All sequential models use **BPRLoss + EmbLoss** (except gSASRec which uses gBCELoss + EmbLoss).

-----

Hybrid Models (4)
=================

Hybrid models combine collaborative filtering signals with side information. All require **side information** to be provided.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Model
     - Hyperparameters
   * - **AddEASE**
     - ``l2`` (float), ``alpha`` (float)
   * - **CEASE**
     - ``l2`` (float), ``alpha`` (float)
   * - **AttributeItemKNN**
     - ``k`` (int), ``similarity`` (str)
   * - **AttributeUserKNN**
     - ``k`` (int), ``similarity`` (str), ``user_profile`` (str)
