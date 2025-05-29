# 💡 [WarpRec](../../README.md) Recommenders

The WarpRec `Recommenders` module offers all the tools needed to train and build your recommendation models. This module contains:

- State-of-the-Art models already implemented for an easy access.
- The trainer compatible with WarpRec models to train your models.
- The building blocks to create your own model.

## 📚 Table of Contents
- 🤖 [Recommendation models](#🤖-recommendation-models)
    - 🌐 [General Recommenders](#🌐-general-recommenders)
- 🏋️‍♂️ [Trainer](#️🏋️‍♂️️-trainer)
- 🧱 [Layers](#🧱-layers)
- 📉 [Losses](#📉-losses)
- 🤝 [Similarities](#🤝-similarities)

## 🤖 Recommendation models

The `Recommendation models` module defines the core abstractions and shared utilities for implementing recommendation algorithms within the WarpRec framework. At its foundation lies the `Recommender` abstract class, which establishes a standardized interface and behavior expected from any model integrated into the system. This includes methods for training (`fit`), generating predictions (`predict`), and producing ranked recommendations (`get_recs`). It also handles common functionalities such as device management, seed initialization for reproducibility, and dynamic parameter loading.

The module is designed to support a variety of recommendation approaches, from neural-based to memory-based models. The base class inherits from `torch.nn.Module`, ensuring compatibility with PyTorch’s ecosystem and enabling easy extensibility for models requiring backpropagation.

To aid development, the module includes utility functions such as matrix normalization and top-k filtering, which are frequently needed in collaborative filtering scenarios. A concrete implementation, `ItemSimRecommender`, is also provided as a reference. This class models recommendations using item-to-item similarities and performs predictions via a simple matrix multiplication (`X @ S`), where `S` is a learnable similarity matrix. It serves as a good starting point for non-neural models that rely on similarity metrics rather than gradient-based learning.

Model-specific attributes are handled dynamically using Python annotations and parameter dictionaries, making it easy to extend or introspect models with minimal boilerplate. Additionally, naming utilities ensure consistent tracking of models and their configurations across experiments.

### 🌐 General Recommenders

The `General Recommenders` module of WarpRec is a collection of `collaborative` and `content-based` models. For further information check the [General Recommenders](general_recommender/README.md) documentation.

## 🏋️‍♂️ Trainer

The WarpRec `Trainer` module provides a powerful and flexible interface for training recommendation models and performing hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

- This module is designed to handle every aspect of the training pipeline:
- Supports both custom parameter configurations and YAML-based configurations via the Configuration object.
- Compatible with all models built using the WarpRec framework.
- Integrates seamlessly with external dashboards and experiment trackers like:
    - [Weights & Biases (wandb)](https://wandb.ai/site/)
    - [CodeCarbon](https://codecarbon.io/)
    - [MLflow](https://mlflow.org/)
- Leverages modern hyperparameter search strategies and schedulers, thanks to a unified wrapper system for optimization.
- Automatically manages GPU/CPU allocation for each trial.

The main entrypoint is the `Trainer` class. After instantiating it with a model name, dataset, and metric, you can call `train_and_evaluate()` to:

1. Launch hyperparameter tuning using Ray Tune.
2. Evaluate each candidate using your selected metric and cutoff.
3. Retrieve the best performing model and its associated parameters.

This design ensures maximum reproducibility, scalability, and extensibility, making it suitable for both academic research and industrial deployment.

## 🧱 Layers

WarpRec `Layers` module offers a series of building blocks that can be used inside your recommendation models. As of right now, the implemented layers inside the module are as follows:

- `MLP`

## 📉 Losses

WarpRec `Losses` module offers a series of loss functions that can be used inside your recommendation models. As of right now, the implemented loss functions inside the module are as follows:

- `BPRLoss`

## 🤝 Similarities

WarpRec `Similarities` module offers a series of similarity functions that can be used inside your recommendation models. As of right now, the implemented similarity functions inside the module are as follows:

- `Cosine`
- `Dot`
- `Euclidean`
- `Manhattan`
- `Haversine`
