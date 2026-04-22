# Collaborative-Filtering Recommenders

The **Collaborative-Filtering Recommenders** module of WarpRec is a collection of collaborative models. In the following sections you will find the list of available models within WarpRec, together with their respective parameters. These models can be used as-is or customized to fit experimental needs.

!!! info "API Reference"

    For class signatures, parameters, and source code, see the [Collaborative Filtering API Reference](../api-reference/recommenders/collaborative.md).

## Summary of Available Models

| Category | Model | Description |
|---|---|---|
| Autoencoders | [EASE](#ease) | Linear autoencoder using ridge regression for item similarity. |
| | [ELSA](#elsa) | Scalable EASE approximation using sparse low-rank decomposition via SGD. |
| | [CDAE](#cdae) | Denoising autoencoder with user-specific latent vectors. |
| | [MacridVAE](#macridvae) | Disentangled VAE modeling macro concepts for user intentions. |
| | [MultiDAE](#multidae) | Denoising autoencoder optimized for implicit data. |
| | [MultiVAE](#multivae) | Variational autoencoder modeling uncertainty in preferences. |
| | [SANSA](#sansa) | Scalable autoencoder using sparse matrix approximations and LDLT decomposition. |
| Graph Based | [DGCF](#dgcf) | Disentangles embeddings into latent factors using iterative routing. |
| | [EGCF](#egcf) | Embedding-less graph model using contrastive learning. |
| | [ESIGCF](#esigcf) | Simplified JoGCN with intent-aware contrastive learning. |
| | [GCMC](#gcmc) | Graph autoencoder for explicit feedback using multi-relational convolutions. |
| | [LightCCF](#lightccf) | Contrastive model with Neighborhood Aggregation loss (supports MF/GCN). |
| | [LightGCL](#lightgcl) | Contrastive learning using SVD for global view augmentation. |
| | [LightGCN](#lightgcn) | Simplified Graph convolutional neural network. |
| | [LightGCN++](#lightgcnpp) | Improved LightGCN with asymmetric normalization and residual connections. |
| | [LightGODE](#lightgode) | Training-free graph convolution using post-training ODE solver. |
| | [MACRGCN](#macrgcn) | LightGCN backbone with counterfactual reasoning for popularity debiasing. |
| | [MixRec](#mixrec) | Dual mixing data augmentation with contrastive learning. |
| | [NGCF](#ngcf) | Complex Graph convolutional neural network. |
| | [PAAC](#paac) | Popularity-aware alignment and contrast for debiasing with LightGCN. |
| | [PopDCL](#popdcl) | Popularity-aware debiased contrastive loss with LightGCN encoder. |
| | [RecDCL](#recdcl) | Dual contrastive learning combining feature-wise and batch-wise objectives. |
| | [RP3Beta](#rp3beta) | Random walk model with popularity penalization. |
| | [SGCL](#sgcl) | Unified supervised contrastive learning without negative sampling. |
| | [SimGCL](#simgcl) | Graph contrastive learning via noise perturbation without graph augmentation. |
| | [SimRec](#simrec) | Graph-less collaborative filtering via contrastive knowledge distillation. |
| | [SGL](#sgl) | Self-supervised learning with graph structure augmentation (ED, ND, RW). |
| | [UltraGCN](#ultragcn) | Efficient GCN approximation using constraint losses without message passing. |
| | [XSimGCL](#xsimgcl) | Graph contrastive learning with noise perturbation. |
| KNN | [ItemKNN](#itemknn) | Item-based collaborative KNN using similarity metrics. |
| | [ItemKNN-TD](#itemknn-td) | Item-based KNN with exponential temporal decay on interactions. |
| | [UserKNN](#userknn) | User-based collaborative KNN using historical interactions. |
| | [UserKNN-TD](#userknn-td) | User-based KNN with exponential temporal decay on interactions. |
| Latent Factor | [ADMMSlim](#admmslim) | Sparse item similarity model optimized via ADMM. |
| | [BPR](#bpr) | Pairwise ranking model for implicit feedback. |
| | [FISM](#fism) | Efficient item similarity model using weighted average as user embeddings. |
| | [MACRMF](#macrmf) | Matrix factorization with counterfactual reasoning for popularity debiasing. |
| | [SLIM](#slim) | Interpretable item similarity model with L1/L2 regularization. |
| Neural | [ConvNCF](#convncf) | Applies CNNs to user-item embeddings outer product to capture structured interaction patterns. |
| | [NeuMF](#neumf) | Hybrid neural model combining GMF and MLP layers. |

## Autoencoders

Autoencoder models learn compact latent representations of users or items by reconstructing user-item interaction data. These models are particularly effective in sparse recommendation settings.

### EASE

EASE (Embarrassingly Shallow Autoencoder): A simple, closed-form linear model that uses ridge regression to learn item-item similarities. Highly efficient and effective as a collaborative filtering baseline.

For further details, please refer to the [paper](https://arxiv.org/abs/1905.03375).

```yaml
models:
  EASE:
    l2: 10
```

### ELSA

ELSA (Efficient Linear Sparse Autoencoder): ELSA is a scalable approximation of the EASE algorithm that replaces the computationally expensive $O(I^3)$ matrix inversion with a sparse, low-rank decomposition optimized via stochastic gradient descent. This allows it to deliver EASE-level recommendation quality for massive item catalogs while significantly reducing memory usage and training time.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3523227.3551482).

```yaml
models:
  ELSA:
    n_dims: 64
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### CDAE

CDAE (Collaborative Denoising Auto-Encoder): A denoising autoencoder that specifically incorporates a user-specific latent vector (bias) into the hidden layer. This allows the model to capture user-specific patterns more effectively than standard autoencoders, making it highly effective for top-N recommendation tasks.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/2835776.2835837).

```yaml
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
```

### MacridVAE

MacridVAE (Macro-Disentangled Variational Autoencoder): A disentangled representation learning model that assumes user intentions are driven by a few macro concepts. It uses a VAE architecture with a specific encoder to separate these high-level concepts, improving interpretability and robustness.

For further details, please refer to the [paper](https://arxiv.org/abs/1910.14238).

```yaml
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
```

### MultiDAE

MultiDAE (Multinomial Denoising Autoencoder): A deep autoencoder trained with dropout for denoising input data. Learns robust latent representations from implicit feedback using a multinomial loss.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3178876.3186150).

```yaml
models:
  MultiDAE:
    intermediate_dim: 600
    latent_dim: 200
    corruption: 1.0
    weight_decay: 0.0001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### MultiVAE

MultiVAE (Multinomial Variational Autoencoder): A probabilistic variant of MultiDAE that models uncertainty in user preferences via variational inference. Useful for capturing diverse user behaviors and providing more personalized recommendations.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3178876.3186150).

```yaml
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
```

### SANSA

SANSA (Scalable Approximate NonSymmetric Autoencoder): SANSA is a collaborative filtering algorithm designed to handle massive datasets by bypassing the memory bottlenecks of traditional linear models through sparse matrix approximations and an $LDL^T$ decomposition. It enables the training of high-performance autoencoders on a single machine, even with millions of items, by maintaining a compact, end-to-end sparse architecture.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3604915.3608827).

!!! warning

    This model to correctly compile requires additional dependencies to run the scikit-sparse module. You can find more information and guides [here](https://github.com/scikit-sparse/scikit-sparse). In case these dependencies are not correctly installed, the model will compute the normal inverse of the target matrix, resulting in different results and slower training times.

```yaml
models:
  SANSA:
    l2: 10
    target_density: 0.01
```

---

## Graph Based

Graph-based recommenders exploit the structure of the user-item interaction graph to infer relationships and make recommendations. These models capture high-order proximity and implicit associations through walks or neighborhood propagation. They are well-suited for uncovering complex patterns in sparse datasets.

!!! warning

    Graph-based models require PyTorch Geometric (PyG) dependencies to be installed correctly. Check the [installation guide](../get-started/installation.md) for more information on how to install them.

### DGCF

DGCF (Disentangled Graph Collaborative Filtering): A graph-based model that disentangles user and item embeddings into multiple latent intents (factors) using an iterative routing mechanism. It encourages independence between factors via a distance correlation loss.

For further details, please refer to the [paper](https://arxiv.org/abs/2007.01764).

```yaml
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
```

### EGCF

EGCF (Embedding-Less Graph Collaborative Filtering): A simplified graph model that removes user embeddings, learning only item embeddings to reduce complexity. It employs a joint loss combining BPR and contrastive learning (InfoNCE) to ensure alignment and uniformity without data augmentation. Supports 'parallel' and 'alternating' propagation modes.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3701230).

```yaml
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
```

### ESIGCF

ESIGCF (Extremely Simplified but Intent-enhanced Graph Collaborative Filtering): A simplified graph model that removes explicit user embeddings and utilizes Joint Graph Convolution (JoGCN) with hybrid normalization. It integrates intent-aware contrastive learning to capture user intents without requiring data augmentation.

For further details, please refer to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625025266).

```yaml
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
```

### GCMC

GCMC (Graph Convolutional Matrix Completion): A graph autoencoder designed for explicit feedback. It treats different rating values as distinct edge types in the user-item graph and learns embeddings using a graph convolutional encoder. A decoder then predicts rating probabilities. **This model requires explicit ratings to function properly**.

For further details, please refer to the [paper](https://arxiv.org/abs/1706.02263).

```yaml
models:
  GCMC:
    embedding_size: 64
    reg_weight: 0.001
    weight_decay: 0.0001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### LightCCF

LightCCF (Light Contrastive Collaborative Filtering): A contrastive learning model that introduces a Neighborhood Aggregation (NA) loss. It brings users closer to their interacted items while pushing them away from other positive pairs (users and items) in the batch. It can work with a standard MF encoder (n_layers=0) or a GCN encoder.

For further details, please refer to the [paper](https://arxiv.org/abs/2504.10113).

```yaml
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
```

### LightGCL

LightGCL (Simple Yet Effective Graph Contrastive Learning): A graph contrastive learning model that uses Singular Value Decomposition (SVD) to construct a global contrastive view. It contrasts the local graph view (GCN) with the global SVD view to enhance representation learning and robustness against noise.

For further details, please refer to the [paper](https://arxiv.org/abs/2302.08191).

```yaml
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
```

### LightGCN

LightGCN: A simplified graph convolutional network designed for collaborative filtering. It eliminates feature transformations and nonlinear activations, focusing solely on neighborhood aggregation.

For further details, please refer to the [paper](https://arxiv.org/abs/2002.02126).

```yaml
models:
  LightGCN:
    embedding_size: 64
    n_layers: 3
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### LightGCN++ { #lightgcnpp }

LightGCN++: An enhanced version of LightGCN that introduces asymmetric normalization (controlled by alpha and beta) and a residual connection to the initial embeddings (controlled by gamma). This allows the model to better adapt to the specific structural properties of the dataset.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3640457.3688176).

```yaml
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
```

### LightGODE

LightGODE (Light Post-Training Graph-ODE): A highly efficient model that trains embeddings without graph convolution using alignment and uniformity losses. It applies a continuous Graph-ODE solver only during inference to incorporate high-order connectivity.

For further details, please refer to the [paper](https://arxiv.org/abs/2407.18910).

```yaml
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
```

### MACRGCN

MACRGCN (Model-Agnostic Counterfactual Reasoning with GCN): Extends the MACR framework by replacing the Matrix Factorization backbone with LightGCN. It maintains a three-branch architecture with a main matching branch, an item module (capturing popularity bias), and a user module (capturing user conformity). During training the branches are fused with a multi-task BCE loss; during inference, counterfactual reasoning removes the direct effect of item popularity on ranking scores (TIE = TE - NDE).

For further details, please refer to the [paper](https://arxiv.org/abs/2010.15363).

```yaml
models:
  MACRGCN:
    embedding_size: 64
    n_layers: 3
    reg_weight: 0.001
    alpha: 0.1
    beta: 0.1
    c: 0.1
    user_mlp_hidden: 64
    item_mlp_hidden: 64
    neg_samples: 1
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### MixRec

MixRec (Individual and Collective Mixing): A graph-based model that employs dual mixing strategies (Individual and Collective) to augment embeddings. It uses a dual-mixing contrastive learning objective to enhance consistency between positive pairs while leveraging mixed negatives.

For further details, please refer to the [paper](https://doi.org/10.1145/3696410.3714565).

```yaml
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
```

### NGCF

NGCF (Neural Graph-based Collaborative Filtering): A neural graph-based collaborative filtering model that explicitly captures high-order connectivity by propagating embeddings through the user-item interaction graph.

For further details, please refer to the [paper](https://arxiv.org/abs/1905.08166).

```yaml
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
```

### PAAC

PAAC (Popularity-Aware Alignment and Contrast): A debiasing model built on a LightGCN backbone with two complementary modules. The Supervised Alignment Module pulls together representations of popular and unpopular co-interacted items, injecting supervision into sparse unpopular embeddings. The Re-weighting Contrast Module splits items into popular and unpopular groups and applies asymmetric negative weighting in InfoNCE losses to prevent excessive separation between groups.

For further details, please refer to the [paper](https://arxiv.org/abs/2405.20718).

```yaml
models:
  PAAC:
    embedding_size: 64
    n_layers: 3
    lambda1: 0.1
    lambda2: 0.1
    temperature: 0.1
    gamma: 0.1
    beta: 0.1
    pop_ratio: 0.1
    eps: 0.001
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### PopDCL

PopDCL (Popularity-aware Debiased Contrastive Loss): A contrastive learning model using a LightGCN encoder that simultaneously corrects positive and negative scores based on popularity. The M+ correction reduces scores of positive pairs likely to be false-positives due to popularity bias, while the M- correction personalizes the debiased contrastive loss using per-user false-negative probabilities. Both corrections rely solely on pre-computed item/user popularity from the training set.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3583780.3615009).

```yaml
models:
  PopDCL:
    embedding_size: 64
    n_layers: 3
    temperature: 0.2
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### RecDCL

RecDCL (Dual Contrastive Learning for Recommendation): Combines Feature-wise Contrastive Learning (FCL) and Batch-wise Contrastive Learning (BCL) on a LightGCN encoder. FCL includes a Barlow-Twins-style cross-correlation loss (UIBT) to reduce inter-user/item redundancy and a polynomial-kernel uniformity loss (UUII) within user/item embeddings. BCL uses historical-embedding augmentation inspired by SimSiam for robust output representations.

For further details, please refer to the [paper](https://arxiv.org/abs/2401.15635).

```yaml
models:
  RecDCL:
    embedding_size: 64
    n_layers: 2
    gamma: 0.01
    alpha: 0.2
    poly_a: 1.0
    poly_c: 0.0000001
    poly_e: 4
    beta: 5.0
    tau_momentum: 0.1
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### RP3Beta

RP3Beta: A graph-based collaborative filtering model that performs a biased random walk of length 3 on the user-item bipartite graph.

For further details, please refer to the [paper](https://www.zora.uzh.ch/id/eprint/131338/1/TiiS_2016.pdf).

```yaml
models:
  RP3Beta:
    k: 10
    alpha: 0.1
    beta: 0.1
    normalize: True
```

### SGCL

SGCL (Supervised Graph Contrastive Learning): A unified framework that merges the recommendation task and self-supervised learning into a single supervised contrastive loss. It simplifies the training pipeline by removing the need for negative sampling and data augmentation.

For further details, please refer to the [paper](https://arxiv.org/abs/2507.13336).

```yaml
models:
  SGCL:
    embedding_size: 64
    n_layers: 3
    temperature: 0.1
    reg_weight: 0.0001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### SimGCL

SimGCL (Simple Graph Contrastive Learning): A graph contrastive learning model that completely discards graph augmentations (such as edge/node dropout). Instead, it creates contrastive views by adding uniform random noise to node embeddings at each GCN layer. Two independently perturbed views are generated per forward pass; an InfoNCE loss maximizes agreement between same-node representations across views, while a BPR loss drives the recommendation task.

For further details, please refer to the [paper](https://arxiv.org/abs/2112.08679).

```yaml
models:
  SimGCL:
    embedding_size: 64
    n_layers: 3
    lambda_: 0.2
    eps: 0.1
    temperature: 0.2
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### SimRec

SimRec (Graph-less Collaborative Filtering): A knowledge distillation framework that trains a lightweight MLP student to match a GCN teacher, eliminating the need for graph convolution at inference time. It employs prediction-level KD, embedding-level contrastive KD, and adaptive contrastive regularization to distill high-order collaborative semantics while mitigating over-smoothing effects from the GNN teacher.

For further details, please refer to the [paper](https://arxiv.org/abs/2303.08537).

```yaml
models:
  SimRec:
    embedding_size: 64
    n_teacher_layers: 3
    n_student_layers: 2
    teacher_reg_weight: 0.001
    lambda1: 0.1
    lambda2: 0.1
    lambda3: 0.1
    lambda4: 0.1
    tau1: 0.1
    tau2: 0.1
    tau3: 0.1
    eps: 0.1
    batch_size_kd: 2048
    teacher_epochs: 200
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
    teacher_learning_rate: 0.001
```

### SGL

SGL (Self-supervised Graph Learning): A graph-based model that augments the user-item graph structure (via Node Dropout, Edge Dropout, or Random Walk) to create auxiliary views for contrastive learning, improving robustness and accuracy.

For further details, please refer to the [paper](https://arxiv.org/abs/2010.10783).

```yaml
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
```

### UltraGCN

UltraGCN: A simplified GCN model that skips explicit message passing during training. It approximates infinite-layer graph convolutions using a constraint loss objective that models both user-item and item-item relationships, resulting in high efficiency and scalability.

For further details, please refer to the [paper](https://arxiv.org/abs/2110.15114).

```yaml
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
```

### XSimGCL

XSimGCL: A graph contrastive learning model that simplifies graph augmentations by adding uniform noise to embeddings. It achieves state-of-the-art performance by regulating the uniformity of the learned representation.

For further details, please refer to the [paper](https://arxiv.org/abs/2209.02544).

```yaml
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
```

---

## KNN (K Nearest Neighbor)

KNN-based models generate recommendations by identifying the most similar users or items based on interaction patterns or side information.

### ItemKNN

ItemKNN: A collaborative item-based KNN model that recommends items similar to those the user has already interacted with.

For further details, please refer to the [paper](http://ieeexplore.ieee.org/document/1167344/).

```yaml
models:
  ItemKNN:
    k: 10
    similarity: cosine
```

### ItemKNN-TD { #itemknn-td }

ItemKNN-TD (Item KNN with Temporal Decay): Extends the standard ItemKNN by applying exponential temporal decay to interaction weights, so that older interactions contribute less to the item-item similarity computation. A higher beta means older interactions decay faster.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/1099554.1099689).

```yaml
models:
  ItemKNN-TD:
    k: 10
    similarity: cosine
    beta: 0.5
```

### UserKNN

UserKNN: A collaborative user-based KNN model that recommends items liked by similar users.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/192844.192905).

```yaml
models:
  UserKNN:
    k: 10
    similarity: cosine
```

### UserKNN-TD { #userknn-td }

UserKNN-TD (User KNN with Temporal Decay): User-side variant of ItemKNN-TD that applies exponential temporal decay to user-user similarity computations, giving less weight to older interactions. A higher beta means older interactions decay faster.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/1099554.1099689).

```yaml
models:
  UserKNN-TD:
    k: 10
    similarity: cosine
    beta: 0.5
```

---

## Latent Factor

Latent factor recommenders decompose the user-item interaction matrix into lower-dimensional representations. These models capture hidden patterns in user preferences and item characteristics, allowing for effective personalization. They include factorization-based approaches, pairwise ranking models, and sparse linear methods that emphasize interpretability and scalability.

### ADMMSlim

ADMMSlim: An efficient implementation of SLIM using the ADMM optimization algorithm. It learns a sparse item-to-item similarity matrix for the top-N recommendation, balancing interpretability and performance.

For further details, please refer to the [paper](https://doi.org/10.1145/3336191.3371774).

```yaml
models:
  ADMMSlim:
    lambda_1: 0.1
    lambda_2: 0.1
    alpha: 0.2
    rho: 0.35
    it: 10
    positive_only: False
    center_columns: False
```

### BPR

BPR: A pairwise ranking model that optimizes the ordering of items for each user. BPR is particularly effective for implicit feedback and is trained to maximize the margin between positive and negative item pairs.

For further details, please refer to the [paper](https://arxiv.org/abs/1205.2618).

```yaml
models:
  BPR:
    embedding_size: 64
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### FISM

FISM: A recommendation algorithm that models item-to-item similarity by learning latent representations of items. Instead of explicitly learning user embeddings, FISM represents each user as the weighted average of the items they have interacted with, enabling efficient and accurate personalized recommendations.

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/2487575.2487589).

```yaml
models:
  FISM:
    embedding_size: 64
    alpha: 0.1
    split_to: 5
    reg_weight: 0.001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

### MACRMF

MACRMF (Model-Agnostic Counterfactual Reasoning with MF): Implements the MACR framework with a Matrix Factorization backbone. It uses a three-branch architecture: a main MF matching branch, an item module that captures the direct effect of item popularity, and a user module that captures user conformity. During training the branches are jointly optimized with a multi-task BCE loss; during inference, counterfactual reasoning subtracts the natural direct effect (NDE) of popularity, leaving only the true user-item matching signal.

For further details, please refer to the [paper](https://arxiv.org/abs/2010.15363).

```yaml
models:
  MACRMF:
    embedding_size: 64
    alpha: 0.1
    beta: 0.1
    c: 0.1
    reg_weight: 0.001
    batch_size: 2048
    neg_samples: 1
    epochs: 200
    learning_rate: 0.001
```

### Slim

Slim: A collaborative filtering model that learns a sparse item similarity matrix using L1 and L2 regularization. SLIM directly models the relationship between items, making it highly interpretable and effective for top-N recommendation.

For further details, please refer to the [paper](https://ieeexplore.ieee.org/document/6137254).

```yaml
models:
  Slim:
    l1: 0.2
    alpha: 0.1
```

---

## Neural

Neural recommenders leverage deep learning architectures to model complex, non-linear interactions between users and items.

### ConvNCF

ConvNCF: Utilizes the outer product of user and item embeddings to construct a 2D interaction map, which is processed by Convolutional Neural Networks (CNNs) to capture complex and localized patterns in user-item interactions. ConvNCF enhances the expressive power of neural collaborative filtering by modeling structured relationships, making it well-suited for scenarios where fine-grained interaction modeling is critical.

For further details, please refer to the [paper](https://arxiv.org/abs/1808.03912).

```yaml
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
```

### NeuMF

NeuMF: Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) to capture both linear and non-linear user-item interactions. NeuMF is a highly expressive model that can adapt to various patterns in user behavior, making it suitable for both implicit and explicit feedback scenarios.

For further details, please refer to the [paper](https://arxiv.org/abs/1708.05031).

```yaml
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
```
