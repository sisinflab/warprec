# pylint: disable = R0801, E1102
from typing import Any

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from scipy.sparse import coo_matrix
from warprec.recommenders.base_recommender import Recommender


class ProxyRecommender(Recommender):
    """Implementation of a ProxyRecommender, used
    to evaluate a recommendation file from other frameworks.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        recommendation_file (str): Path to the recommendation file.
        separator (str): Separator of the recommendation file.
        header (bool): Wether or not the recommendation file has an header.

    Raises:
        ValueError: If the item and user mappings or the number of items and users are not provided
             or if the recommendation file is malformed.
        FileNotFoundError: If the recommendation file does not exist.
        RuntimeError: If an error occurs while reading the recommendation file.
    """

    recommendation_file: str
    separator: str
    header: bool

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, info=info, *args, **kwargs)

        _imap: dict = info.get("item_mapping", None)
        _umap: dict = info.get("user_mapping", None)
        num_items: int = info.get("items", None)
        num_users: int = info.get("users", None)
        if any(x is None or x == {} for x in [_imap, _umap, num_items, num_users]):
            raise ValueError(
                "Number of items and users must be provided in the info dictionary. "
                "Item and user mapping must be provided to correctly initialize the model."
            )

        try:
            recommendation_df: DataFrame
            if self.header:
                recommendation_df = pd.read_csv(
                    self.recommendation_file,
                    sep=self.separator,
                    dtype={"user_id": int, "item_id": int, "rating": float},
                    usecols=["user_id", "item_id", "rating"],
                )
            else:
                recommendation_df = pd.read_csv(
                    self.recommendation_file, sep=self.separator, header=None
                )
                recommendation_df = recommendation_df.iloc[:, :3]
                recommendation_df.columns = ["user_id", "item_id", "rating"]

                users = recommendation_df["user_id"].map(_umap).values
                items = recommendation_df["item_id"].map(_imap).values
                ratings = recommendation_df["rating"].values

                # Compute invalid values mask for faster filtering
                mask = ~np.isnan(users) & ~np.isnan(items) & ~np.isnan(ratings)

                # Filter out invalid values
                users = users[mask]
                items = items[mask]
                ratings = ratings[mask]

                self._predictions_sparse = coo_matrix(
                    (ratings, (users, items)),
                    shape=(num_users, num_items),
                    dtype=np.float32,
                ).tocsr()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Recommendation file {self.recommendation_file} not found."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Recommendation file {self.recommendation_file} is empty."
            )
        except pd.errors.ParserError:
            raise ValueError(
                f"Recommendation file {self.recommendation_file} is malformed."
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while reading the recommendation file: {e}"
            )

    @torch.no_grad()
    def predict(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Access the scores from the recommendation file
        predictions_numpy = self._predictions_sparse[user_indices.cpu()].toarray()
        predictions = torch.from_numpy(predictions_numpy).to(self._device)

        # Masking interaction already seen in train
        predictions[train_batch != 0] = -torch.inf
        return predictions.to(self._device)
