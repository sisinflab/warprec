import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from elliotwo.data.dataset import AbstractDataset
from elliotwo.utils.config import Configuration
from elliotwo.recommenders.abstract_recommender import ItemSimilarityRecommender


class EASE(ItemSimilarityRecommender):
    """Implementation of EASE algorithm from Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    Attributes:
        VERSION (str): The version of the model.

    Args:
        config (Configuration): The configuration of the experiement.
        dataset (AbstractDataset): The dataset to train the model on.
        params (dict): The parameters of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    The params allowed by this model are as follows:
        l2 (float): Normalization parameter to use during train.
        implementation (str): The type of implementation to use:
            classic: The standard EASE implementation as in the original paper.
            elliot: The Elliot modified implementation with popularity penalization.
    """

    VERSION: str = "0.1"

    def __init__(
        self,
        config: Configuration,
        dataset: AbstractDataset,
        params: dict,
        *args,
        **kwargs,
    ):
        super().__init__(config, dataset, params, *args, **kwargs)

        self.X = self.interaction_matrix
        self.l2 = self._params["l2"]
        self.implementation = self._params["implementation"]
        self._model_version = self.VERSION

    def fit(self):
        """During training we will compute the B similarity matrix {item x item}."""
        if self.implementation == "classic":
            # The classic implementation follows the original paper
            G = self.X.T @ self.X + self.l2 * np.identity(self.X.shape[1])
            B = np.linalg.inv(G)
            B /= -np.diag(B)
            np.fill_diagonal(B, 0.0)

            self.item_similarity = B

        elif self.implementation == "elliot":
            # The 'elliot' implementation add a popularity penalization to the classic definition of the EASE model
            # Gram matrix
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

        else:
            raise ValueError(
                f"EASE model does not support {self.implementation} implementation"
            )
