.. _recommender:

#################
Recommenders
#################

.. toctree::
   :hidden:
   :maxdepth: 2

   collaborative
   content
   hybrid
   context
   sequential
   unpersonalized
   guide/index

WarpRec provides a comprehensive collection of **built-in recommender algorithms**, covering both classical and state-of-the-art approaches widely used in the recommendation systems domain.
These models are designed to be **ready-to-use**, serving as a strong baseline for experimentation, benchmarking, and real-world deployment.

All built-in models are implemented with a focus on **efficiency and scalability**, leveraging **WarpRec’s optimized training backend** to support large-scale datasets and high-performance execution.
Users can flexibly configure architectures and hyperparameters, making it straightforward to adapt each model to specific recommendation scenarios.

Furthermore, WarpRec’s **modular design** makes it easy to extend the framework with **custom recommenders**. By inheriting from the provided base classes and reusing existing components (e.g., interaction handlers, loss modules, evaluation pipelines), researchers and practitioners can rapidly prototype new algorithms while maintaining integration with the training and evaluation ecosystem.

This combination of **ready-to-use models**, **efficient implementations**, and **customization support** ensures that WarpRec can serve both as a practical toolkit for applied recommendation systems and as a flexible research platform for advancing new methodologies.

-----

Algorithm Taxonomy
==================

WarpRec ships with **57 built-in algorithms** spanning 6 model families.
All models can run locally or at cluster scale via Ray.

.. list-table::
   :header-rows: 1
   :widths: 18 15 67

   * - Family
     - Model
     - Description
   * - **Unpersonalized**
     - Pop
     - Recommends the most popular items overall.
   * -
     - Random
     - Recommends items uniformly at random (lower-bound baseline).
   * -
     - ProxyRecommender
     - Evaluates precomputed recommendation lists from an external file.
   * - **Content-Based**
     - VSM
     - Vector Space Model using TF-IDF and cosine similarity on item profiles.
   * - **CF / Autoencoder**
     - EASE
     - Closed-form linear autoencoder via ridge regression for item similarity.
   * -
     - ELSA
     - Scalable EASE approximation using sparse low-rank SGD decomposition.
   * -
     - CDAE
     - Denoising autoencoder with user-specific latent vectors.
   * -
     - MacridVAE
     - Disentangled VAE modeling macro user intentions via concept routing.
   * -
     - MultiDAE
     - Multinomial denoising autoencoder for implicit feedback.
   * -
     - MultiVAE
     - Variational autoencoder with reparameterization for implicit feedback.
   * -
     - SANSA
     - Sparse approximate non-symmetric autoencoder via LDL\ :sup:`T` decomposition.
   * - **CF / Graph-Based**
     - DGCF
     - Disentangled graph CF with iterative factor routing.
   * -
     - EGCF
     - Embedding-less graph CF using BPR + InfoNCE contrastive learning.
   * -
     - ESIGCF
     - Extremely simplified intent-enhanced graph CF with JoGCN.
   * -
     - GCMC
     - Graph convolutional matrix completion for explicit feedback.
   * -
     - LightCCF
     - Contrastive CF with Neighborhood Aggregation loss (MF or GCN encoder).
   * -
     - LightGCL
     - Graph contrastive learning using SVD for global view augmentation.
   * -
     - LightGCN
     - Simplified GCN with linear propagation (no feature transforms).
   * -
     - LightGCN++
     - LightGCN with asymmetric normalization and residual connections.
   * -
     - LightGODE
     - Training-free graph convolution; applies ODE solver at inference.
   * -
     - MixRec
     - Dual individual/collective mixing with contrastive learning.
   * -
     - NGCF
     - Neural graph CF with higher-order connectivity propagation.
   * -
     - RP3Beta
     - Biased random walk of length 3 on the user-item bipartite graph.
   * -
     - SGCL
     - Supervised graph contrastive learning without negative sampling.
   * -
     - SGL
     - Self-supervised graph learning with structure augmentation (ED/ND/RW).
   * -
     - UltraGCN
     - Infinite-layer GCN approximation via constraint losses (no message passing).
   * -
     - XSimGCL
     - Graph contrastive learning with uniform noise perturbation.
   * - **CF / KNN**
     - ItemKNN
     - Item-based collaborative KNN using configurable similarity metrics.
   * -
     - UserKNN
     - User-based collaborative KNN from historical interactions.
   * - **CF / Latent Factor**
     - ADMMSlim
     - Sparse item similarity matrix optimized via ADMM.
   * -
     - BPR
     - Bayesian Personalized Ranking for pairwise implicit feedback.
   * -
     - FISM
     - Factored item similarity with weighted-average user representations.
   * -
     - Slim
     - Sparse linear method with L1/L2 (ElasticNet) regularization.
   * - **CF / Neural**
     - ConvNCF
     - CNN on user-item embedding outer product for structured interaction patterns.
   * -
     - NeuMF
     - Hybrid neural CF combining GMF and MLP branches.
   * - **Context-Aware**
     - AFM
     - Attentional Factorization Machine with attention-weighted feature interactions.
   * -
     - DCN
     - Deep & Cross Network for explicit bounded-degree feature crossing.
   * -
     - DCNv2
     - Improved DCN with Mixture-of-Experts and low-rank cross layers.
   * -
     - DeepFM
     - Parallel FM + DNN for low-order and high-order feature interactions.
   * -
     - FM
     - Factorization Machine modeling second-order feature interactions.
   * -
     - NFM
     - Neural FM with Bi-Interaction pooling layer followed by MLP.
   * -
     - WideAndDeep
     - Joint wide (linear) + deep (DNN) model for memorization and generalization.
   * -
     - xDeepFM
     - Compressed Interaction Network (CIN) for vector-wise explicit interactions.
   * - **Sequential / CNN**
     - Caser
     - Convolutional sequence embedding with horizontal and vertical filters.
   * - **Sequential / Markov**
     - FOSSIL
     - First-order Markov chain fused with factored item similarity.
   * - **Sequential / RNN**
     - GRU4Rec
     - Session-based GRU for short-term preference modeling.
   * -
     - NARM
     - Hybrid GRU encoder with global + local attention mechanisms.
   * - **Sequential / Transformer**
     - BERT4Rec
     - Bidirectional Transformer with masked item prediction (cloze task).
   * -
     - CORE
     - Consistent Representation Encoder unifying encoding/decoding spaces.
   * -
     - gSASRec
     - General self-attention with group-wise binary cross-entropy loss.
   * -
     - LightSANs
     - Low-rank decomposed self-attention with decoupled position encoding.
   * -
     - LinRec
     - Linear attention mechanism (O(N)) for efficient long-sequence modeling.
   * -
     - SASRec
     - Self-attentive Transformer for sequential recommendation.
   * - **Hybrid / Autoencoder**
     - AddEASE
     - EASE extension solving two linear problems to incorporate side information.
   * -
     - CEASE
     - Closed-form extended EASE via augmented interaction matrix with side info.
   * - **Hybrid / KNN**
     - AttributeItemKNN
     - Item-based KNN using content features for similarity.
   * -
     - AttributeUserKNN
     - User-based KNN using content-derived user profiles.
