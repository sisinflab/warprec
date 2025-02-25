# pylint: disable=invalid-name
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from elliotwo.data.dataset import AbstractDataset
from elliotwo.utils.config import Configuration
from elliotwo.recommenders.abstract_recommender import ItemSimilarityRecommender
from elliotwo.utils.registry import model_registry


class EASE(ItemSimilarityRecommender):
    """The main class for EASE models.

    Main definition of attributes and data
    preparation shared between all implementations.

    Args:
        config (Configuration): The configuration of the experiement.
        dataset (AbstractDataset): The dataset to train the model on.
        params (dict): The parameters of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        config: Configuration,
        dataset: AbstractDataset,
        params: dict,
        *args,
        **kwargs,
    ):
        super().__init__(config, dataset, params, *args, **kwargs)

        self._name = "EASE"
        self.X = self.interaction_matrix
        self.l2 = params["l2"]


@model_registry.register(name="EASE")
class EASE_Classic(EASE):
    """Implementation of EASE algorithm from Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    The params allowed by this model are as follows:
        l2 (float): Normalization parameter to use during train.
    """

    def fit(self):
        """During training we will compute the B similarity matrix {item x item}."""
        # The classic implementation follows the original paper
        G = self.X.T @ self.X + self.l2 * np.identity(self.X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B


@model_registry.register(name="EASE", implementation="Elliot")
class EASE_Elliot(EASE):
    """Implementation of EASE algorithm from Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    This implementation was revised in the original Elliot framework.

    The params allowed by this model are as follows:
        l2 (float): Normalization parameter to use during train.
    """

    def fit(self):
        """During training we will compute the B similarity matrix {item x item}."""
        # The 'elliot' implementation add a popularity penalization
        self.item_similarity = safe_sparse_dot(self.X.T, self.X, dense_output=True)

        # Find diagonal indices and define item popularity to use as penalization
        diagonal_indices = np.diag_indices(self.item_similarity.shape[0])
        item_popularity = np.ediff1d(self.X.tocsc().indptr)

        # Penalize item on the diagonal with l2 norm and popularity
        self.item_similarity[diagonal_indices] = item_popularity + self.l2

        # Inverse and normalization
        P = np.linalg.inv(self.item_similarity)
        self.item_similarity = P / (-np.diag(P))

        # Remove diagonal items as in the calssic implementation
        self.item_similarity[diagonal_indices] = 0.0
