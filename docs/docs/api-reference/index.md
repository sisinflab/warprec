# API Reference

This section contains the auto-generated API documentation for WarpRec's core components. Each page provides class signatures, parameters, attributes, and source code for the corresponding module.

## Data

- [Dataset](data/dataset.md) -- Dataset and evaluation dataloaders.
- [Entities](data/entities.md) -- Interaction, Session and training dataloaders.
- [Filtering](data/filtering.md) -- Filter.
- [Reader](data/reader.md) -- Reader, ReaderFactory and Reader implementations.
- [Splitting](data/splitting.md) -- Splitter and SplittingStrategies.
- [Writer](data/writer.md) -- Writer, WriterFactory and Writer implementations.

## Recommenders

- [Model Interfaces](recommenders/base.md) -- Model interfaces and Mixin classes.
- [Collaborative Filtering](recommenders/collaborative.md) -- Autoencoder, Graph-Based, KNN, Latent Factor, and Neural models.
- [Content-Based](recommenders/content.md) -- Vector Space Model and content-based approaches.
- [Hybrid](recommenders/hybrid.md) -- Hybrid autoencoder and KNN models.
- [Context-Aware](recommenders/context.md) -- Factorization Machine variants and deep context models.
- [Sequential](recommenders/sequential.md) -- CNN, RNN, Markov, and Transformer-based sequential models.
- [Unpersonalized](recommenders/unpersonalized.md) -- Popularity and Random baselines.
- [Proxy](recommenders/proxy.md) -- Proxy Recommender for cross-framework evaluation.

## Metrics

- [Metric Interfaces](metrics/base.md) -- Metric interfaces and utility classes.
- [Accuracy](metrics/accuracy.md) -- AUC, F1, GAUC, HitRate, LAUC, MAP, MAR, MRR, nDCG, Precision, Recall.
- [Bias](metrics/bias.md) -- ACLT, APLT, ARP, PopREO, PopRSP.
- [Coverage](metrics/coverage.md) -- ItemCoverage, UserCoverage, NumRetrieved, UserCoverageAtN.
- [Diversity](metrics/diversity.md) -- Gini, ShannonEntropy, SRecall.
- [Fairness](metrics/fairness.md) -- BiasDisparity, ItemMAD, REO, RSP, UserMAD.
- [Novelty](metrics/novelty.md) -- EFD, EPC.
- [Rating](metrics/rating.md) -- MAE, MSE, RMSE.
- [Multiobjective](metrics/multiobjective.md) -- EucDistance, Hypervolume.
