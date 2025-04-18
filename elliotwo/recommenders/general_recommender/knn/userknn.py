# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
from torch import Tensor, nn
from scipy.sparse import csr_matrix
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry, similarities_registry


@model_registry.register(name="UserKNN")
class UserKNN(Recommender):
    """Implementation of UserKNN algorithm from
        GroupLens: an open architecture for collaborative filtering of netnews 1994.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        normalize (bool): Wether or not to normalize the interactions.
    """

    k: int
    similarity: str
    normalize: bool

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
        self._name = "UserKNN"
        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.user_similarity = nn.Parameter(torch.rand(users, users))

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions.
        During the train a similarity matrix {user x user} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Apply normalization of interactions if requested
        if self.normalize:
            X = self._normalize(X)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity with a new nn.Parameter
        self.user_similarity = nn.Parameter(filtered_sim_matrix)

        if report_fn is not None:
            report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The interactions matrix
                that will be used to predict.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])
        r = (
            self.user_similarity.detach().numpy()[start_idx:end_idx, start_idx:end_idx]
            @ interaction_matrix
        )

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """
