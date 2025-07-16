# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from scipy.sparse import csr_matrix, coo_matrix
from warprec.data.dataset import Interactions
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
        self._name = "ProxyRecommender"

        self._imap = info.get("item_mapping", None)
        self._umap = info.get("user_mapping", None)
        self.items = info.get("items", None)
        self.users = info.get("users", None)
        if any(
            x is None or x == {}
            for x in [self._imap, self._umap, self.items, self.users]
        ):
            raise ValueError(
                "Number of items and users must be provided in the info dictionary. "
                "Item and user mapping must be provided to correctly initialize the model."
            )

        try:
            self.recommendation_df: DataFrame
            if self.header:
                self.recommendation_df = pd.read_csv(
                    self.recommendation_file,
                    sep=self.separator,
                    dtype={"user_id": int, "item_id": int, "rating": float},
                    usecols=["user_id", "item_id", "rating"],
                )
            else:
                self.recommendation_df = pd.read_csv(
                    self.recommendation_file, sep=self.separator, header=None
                )
                self.recommendation_df = self.recommendation_df.iloc[:, :3]
                self.recommendation_df.columns = ["user_id", "item_id", "rating"]
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

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """

        users = self.recommendation_df["user_id"].map(self._umap).values
        items = self.recommendation_df["item_id"].map(self._imap).values
        ratings = self.recommendation_df["rating"].values

        # Compute invalid values mask for faster filtering
        mask = ~np.isnan(users) & ~np.isnan(items) & ~np.isnan(ratings)

        # Filter out invalid values
        users = users[mask]
        items = items[mask]
        ratings = ratings[mask]

        self._predictions_sparse = coo_matrix(
            (ratings, (users, items)), shape=(self.users, self.items), dtype=np.float32
        ).tocsr()

        if report_fn is not None:
            report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])

        # Access the scores from the recommendation file
        r = self._predictions_sparse[start_idx:end_idx].toarray()

        # Mask the non-zero entries and convert to tensor
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        pass
