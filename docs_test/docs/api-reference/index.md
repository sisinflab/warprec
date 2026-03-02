# API Reference

This section contains the auto-generated API documentation for WarpRec's core components. Each page provides class signatures, parameters, attributes, and source code for the corresponding module.

## Data Management

- [Data Management](data-management.md) -- Reader, Writer, Dataset, and Splitter classes.

## Recommenders

- [Collaborative Filtering](recommenders/collaborative.md) -- Autoencoder, Graph-Based, KNN, Latent Factor, and Neural models.
- [Content-Based](recommenders/content.md) -- Vector Space Model and content-based approaches.
- [Hybrid](recommenders/hybrid.md) -- Hybrid autoencoder and KNN models.
- [Context-Aware](recommenders/context.md) -- Factorization Machine variants and deep context models.
- [Sequential](recommenders/sequential.md) -- CNN, RNN, Markov, and Transformer-based sequential models.
- [Unpersonalized](recommenders/unpersonalized.md) -- Popularity, Random, and ProxyRecommender baselines.

## Metrics

- [Accuracy](metrics/accuracy.md) -- AUC, F1, GAUC, HitRate, LAUC, MAP, MAR, MRR, nDCG, Precision, Recall.
- [Bias](metrics/bias.md) -- ACLT, APLT, ARP, PopREO, PopRSP.
- [Coverage](metrics/coverage.md) -- ItemCoverage, UserCoverage, NumRetrieved, UserCoverageAtN.
- [Diversity](metrics/diversity.md) -- Gini, ShannonEntropy, SRecall.
- [Fairness](metrics/fairness.md) -- BiasDisparity, ItemMAD, REO, RSP, UserMAD.
- [Novelty](metrics/novelty.md) -- EFD, EPC.
- [Rating](metrics/rating.md) -- MAE, MSE, RMSE.
- [Multiobjective](metrics/multiobjective.md) -- EucDistance, Hypervolume.
