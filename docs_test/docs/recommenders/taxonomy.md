# Models Taxonomy

WarpRec ships with **50+ built-in algorithms** spanning 6 model families. All models can run locally or at cluster scale via Ray.

| Family | Model | Description |
|---|---|---|
| **Unpersonalized** | [Pop](unpersonalized.md#pop) | Recommends the most popular items overall. |
| | [Random](unpersonalized.md#random) | Recommends items uniformly at random (lower-bound baseline). |
| | [ProxyRecommender](cross-framework-evaluation.md) | Evaluates precomputed recommendation lists from an external file. |
| **Content-Based** | [VSM](content.md#vsm) | Vector Space Model using TF-IDF and cosine similarity on item profiles. |
| **CF / Autoencoder** | [EASE](collaborative.md#ease) | Closed-form linear autoencoder via ridge regression for item similarity. |
| | [ELSA](collaborative.md#elsa) | Scalable EASE approximation using sparse low-rank SGD decomposition. |
| | [CDAE](collaborative.md#cdae) | Denoising autoencoder with user-specific latent vectors. |
| | [MacridVAE](collaborative.md#macridvae) | Disentangled VAE modeling macro user intentions via concept routing. |
| | [MultiDAE](collaborative.md#multidae) | Multinomial denoising autoencoder for implicit feedback. |
| | [MultiVAE](collaborative.md#multivae) | Variational autoencoder with reparameterization for implicit feedback. |
| | [SANSA](collaborative.md#sansa) | Sparse approximate non-symmetric autoencoder via LDL^T decomposition. |
| **CF / Graph-Based** | [DGCF](collaborative.md#dgcf) | Disentangled graph CF with iterative factor routing. |
| | [EGCF](collaborative.md#egcf) | Embedding-less graph CF using BPR + InfoNCE contrastive learning. |
| | [ESIGCF](collaborative.md#esigcf) | Extremely simplified intent-enhanced graph CF with JoGCN. |
| | [GCMC](collaborative.md#gcmc) | Graph convolutional matrix completion for explicit feedback. |
| | [LightCCF](collaborative.md#lightccf) | Contrastive CF with Neighborhood Aggregation loss (MF or GCN encoder). |
| | [LightGCL](collaborative.md#lightgcl) | Graph contrastive learning using SVD for global view augmentation. |
| | [LightGCN](collaborative.md#lightgcn) | Simplified GCN with linear propagation (no feature transforms). |
| | [LightGCN++](collaborative.md#lightgcnpp) | LightGCN with asymmetric normalization and residual connections. |
| | [LightGODE](collaborative.md#lightgode) | Training-free graph convolution; applies ODE solver at inference. |
| | [MixRec](collaborative.md#mixrec) | Dual individual/collective mixing with contrastive learning. |
| | [NGCF](collaborative.md#ngcf) | Neural graph CF with higher-order connectivity propagation. |
| | [RP3Beta](collaborative.md#rp3beta) | Biased random walk of length 3 on the user-item bipartite graph. |
| | [SGCL](collaborative.md#sgcl) | Supervised graph contrastive learning without negative sampling. |
| | [SGL](collaborative.md#sgl) | Self-supervised graph learning with structure augmentation (ED/ND/RW). |
| | [UltraGCN](collaborative.md#ultragcn) | Infinite-layer GCN approximation via constraint losses (no message passing). |
| | [XSimGCL](collaborative.md#xsimgcl) | Graph contrastive learning with uniform noise perturbation. |
| **CF / KNN** | [ItemKNN](collaborative.md#itemknn) | Item-based collaborative KNN using configurable similarity metrics. |
| | [UserKNN](collaborative.md#userknn) | User-based collaborative KNN from historical interactions. |
| **CF / Latent Factor** | [ADMMSlim](collaborative.md#admmslim) | Sparse item similarity matrix optimized via ADMM. |
| | [BPR](collaborative.md#bpr) | Bayesian Personalized Ranking for pairwise implicit feedback. |
| | [FISM](collaborative.md#fism) | Factored item similarity with weighted-average user representations. |
| | [Slim](collaborative.md#slim) | Sparse linear method with L1/L2 (ElasticNet) regularization. |
| **CF / Neural** | [ConvNCF](collaborative.md#convncf) | CNN on user-item embedding outer product for structured interaction patterns. |
| | [NeuMF](collaborative.md#neumf) | Hybrid neural CF combining GMF and MLP branches. |
| **Context-Aware** | [AFM](context.md#afm) | Attentional Factorization Machine with attention-weighted feature interactions. |
| | [DCN](context.md#dcn) | Deep & Cross Network for explicit bounded-degree feature crossing. |
| | [DCNv2](context.md#dcnv2) | Improved DCN with Mixture-of-Experts and low-rank cross layers. |
| | [DeepFM](context.md#deepfm) | Parallel FM + DNN for low-order and high-order feature interactions. |
| | [FM](context.md#fm) | Factorization Machine modeling second-order feature interactions. |
| | [NFM](context.md#nfm) | Neural FM with Bi-Interaction pooling layer followed by MLP. |
| | [WideAndDeep](context.md#wideanddeep) | Joint wide (linear) + deep (DNN) model for memorization and generalization. |
| | [xDeepFM](context.md#xdeepfm) | Compressed Interaction Network (CIN) for vector-wise explicit interactions. |
| **Sequential / CNN** | [Caser](sequential.md#caser) | Convolutional sequence embedding with horizontal and vertical filters. |
| **Sequential / Markov** | [FOSSIL](sequential.md#fossil) | First-order Markov chain fused with factored item similarity. |
| **Sequential / RNN** | [GRU4Rec](sequential.md#gru4rec) | Session-based GRU for short-term preference modeling. |
| | [NARM](sequential.md#narm) | Hybrid GRU encoder with global + local attention mechanisms. |
| **Sequential / Transformer** | [BERT4Rec](sequential.md#bert4rec) | Bidirectional Transformer with masked item prediction (cloze task). |
| | [CORE](sequential.md#core) | Consistent Representation Encoder unifying encoding/decoding spaces. |
| | [gSASRec](sequential.md#gsasrec) | General self-attention with group-wise binary cross-entropy loss. |
| | [LightSANs](sequential.md#lightsans) | Low-rank decomposed self-attention with decoupled position encoding. |
| | [LinRec](sequential.md#linrec) | Linear attention mechanism (O(N)) for efficient long-sequence modeling. |
| | [SASRec](sequential.md#sasrec) | Self-attentive Transformer for sequential recommendation. |
| **Hybrid / Autoencoder** | [AddEASE](hybrid.md#addease) | EASE extension solving two linear problems to incorporate side information. |
| | [CEASE](hybrid.md#cease) | Closed-form extended EASE via augmented interaction matrix with side info. |
| **Hybrid / KNN** | [AttributeItemKNN](hybrid.md#attributeitemknn) | Item-based KNN using content features for similarity. |
| | [AttributeUserKNN](hybrid.md#attributeuserknn) | User-based KNN using content-derived user profiles. |
