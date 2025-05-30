# 💡 [WarpRec](../../README.md) Recommenders

The WarpRec `Recommenders` module offers all the tools needed to train and build your recommendation models. This module contains:

- State-of-the-Art models already implemented for an easy access.
- The trainer compatible with WarpRec models to train your models.
- The building blocks to create your own model.

## 📚 Table of Contents
- 🤖 [Recommendation models](#🤖-recommendation-models)
    - 🌐 [General Recommenders](#🌐-general-recommenders)
    - 👉 [How to implement your recommender](#👉-how-to-implement-your-recommender)
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

### 👉 How to implement Your recommender

In this section we will guide you through the process of implementing you own recommendation model. First of all, let's import the main metric interface:

```python
from warprec.recommenders.base_recommender import ItemSimRecommender, Recommender
```

The **Recommender** interface is gonna be sufficient for any kind of recommender model, **ItemSimRecommender** is a special interface used by models that learn an item similarity matrix.

In this guide we will go through the implementation of the EASE model, which is a really simple model to get you started with WarpRec interfaces. As you know, EASE learns an item similarity matrix, but in this case we will use the base interface **Recommender**.

The first step is the initialization of the model, here we need to setup all the key component needed for the model to execute and all the *learnable objects*. Let's see the key step to a correct initialization:

- Write the parameters of your model as in the example, WarpRec handles parameters through annotation and will populate them through configuration.
- Configure your layers and parameters that will be learned during the fitting of the model.
- Change the name of the model accordingly.

```python
class MyEASE(Recommender):
    l2: float

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.item_similarity = nn.Parameter(torch.rand(self.items, self.items))
        self._name = "EASE"
```

**The info dictionary contains the main information of the dataset**. These are the information contained in it:

- **items**: Number of unique items.
- **users**: Number of unique users.
- **features**: Number of unique features.

After the initialization we will need to implement the three main methods of the model:

1. **.fit()**: The main method of training, this is where you will implement the main trining cycle.
2. **.forward()**: The forward step of the model, used inside neural models.
3. **.predict()**: The prediction of the model, given the training interactions.

As for our example, the second step is not needed because the is no neural layer inside EASE, therefore no backward step either. As for the fit, we can implement it like this:

```python
def fit(
    self,
    interactions: Interactions,
    *args: Any,
    report_fn: Optional[Callable] = None,
    **kwargs: Any,
):
    X = interactions.get_sparse()

    G = X.T @ X + self.l2 * np.identity(X.shape[1])
    B = np.linalg.inv(G)
    B /= -np.diag(B)
    np.fill_diagonal(B, 0.0)

    self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))

    if report_fn is not None:
        report_fn(self)
```

We can access the interaction matrix in sparse format from the interaction object, after that we can proceed with the normal implementation. We also update the learned parameter and, if a reporting function is provided, we report the model state. This is used inside Ray Tune for checkpoint creation and management.

At this point, the last step is to implement the .predict() method. For this model we can implement it as such:

```python
@torch.no_grad()
def predict(
    self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
) -> Tensor:
    r = interaction_matrix @ self.item_similarity.detach().numpy()

    # Masking interaction already seen in train
    r[interaction_matrix.nonzero()] = -torch.inf
    return torch.from_numpy(r).to(self._device)
```

When you implement you model, remember to follow the important step of masking the training interactions and also to convert (if not already) to a tensor, and change the device. And you are done!

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
