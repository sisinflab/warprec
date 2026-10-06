# pylint: disable = R0801, E1102
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from warprec.data.entities import Interactions, KnowledgeGraph, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.knowledge_aware_recommender.knowledge_utils import (
    KnowledgeRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="KaHFM")
class KaHFM(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of KaHFM algorithm from
        How to Make Latent Factors Interpretable by Feeding Factorization
        Machines with Knowledge Graphs (ISWC 2019), extended in Semantic
        Interpretation of Top-N Recommendations (TKDE 2020)

    A factorization model whose factors are not latent: there is one per
    feature the graph states about the items, so a user's factor is how much
    they care about *directed by Kubrick* and an item's is how much it is about
    it. The factors start from what is already known, an item at the TF-IDF of
    its features and a user at the mean of the items they interacted with, and
    BPR refines them from there, which keeps every factor tied to the feature
    it started from.

    The original implementation is in Elliot as KaHFM, KaHFMBatch and
    KaHFMEmbeddings, which differ in how they are optimised; this one trains in
    mini-batches through the configured optimizer. Elliot keeps, for each
    feature of a user's profile, only the value of the last item carrying it;
    the profile here is the mean the paper defines.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        min_feature_items (int): How many items must carry a feature for it to
            become a factor.
        reg_weight (float): The L2 regularization weight of the factors.
        bias_reg_weight (float): The L2 regularization weight of the item biases.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If the interactions were not provided, or no feature is
            carried by at least min_feature_items items.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    min_feature_items: int
    reg_weight: float
    bias_reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        knowledge: Optional[KnowledgeGraph] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__(params, info, *args, knowledge=knowledge, seed=seed, **kwargs)

        if interactions is None:
            raise ValueError(
                "KaHFM starts every user from the items they interacted with, so "
                "it needs the interactions at construction."
            )

        features, labels = self._graph.item_features(
            order=1, min_items=self.min_feature_items
        )
        # Annotated here rather than on the class: class annotations are read
        # as hyperparameters, and a run is named after them.
        self.feature_labels: List[Tuple[Any, ...]] = labels
        if features.size(1) == 0:
            raise ValueError(
                f"No feature of the knowledge graph is carried by "
                f"{self.min_feature_items} items or more, so KaHFM has no factor "
                "to learn. Lower min_feature_items."
            )

        item_factors = self.tfidf(features)
        user_factors = self.profiles(interactions, item_factors)

        # The row past the catalogue is the item a sampled candidate list is
        # padded with, and it stays at zero.
        padding = torch.zeros(1, item_factors.size(1))
        self.user_factors = nn.Embedding.from_pretrained(user_factors, freeze=False)
        self.item_factors = nn.Embedding.from_pretrained(
            torch.cat([item_factors, padding]), freeze=False, padding_idx=self.n_items
        )
        self.item_bias = nn.Embedding(self.n_items + 1, 1, padding_idx=self.n_items)
        nn.init.zeros_(self.item_bias.weight)

        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    @staticmethod
    def tfidf(features: Tensor) -> Tensor:
        """The TF-IDF of every item over its features, at unit length.

        A feature is either carried or not, so its term frequency is one and
        its weight is the inverse document frequency ``ln(N / df)``, where N is
        the number of items that carry any feature. An item with no feature,
        or only features every item carries, is left at zero.

        Args:
            features (Tensor): The sparse binary {item x feature} matrix.

        Returns:
            Tensor: The dense {item x feature} TF-IDF matrix, one unit row per item.
        """
        rows, columns = features.coalesce().indices()
        n_items, n_features = features.shape

        frequency = torch.bincount(columns, minlength=n_features).float()
        described = torch.unique(rows).numel()
        values = torch.log(described / frequency.clamp(min=1))[columns]

        norm = torch.zeros(n_items).index_add_(0, rows, values**2).sqrt()
        values = values / norm[rows].clamp(min=torch.finfo(values.dtype).tiny)

        dense = torch.zeros(n_items, n_features)
        dense[rows, columns] = values
        return dense

    def profiles(self, interactions: Interactions, item_factors: Tensor) -> Tensor:
        """Where every user starts: the mean of the items they interacted with.

        The mean is taken over the whole history, items the graph says nothing
        about included, as the paper defines the profile.

        Args:
            interactions (Interactions): The training interactions.
            item_factors (Tensor): The {item x feature} starting factors.

        Returns:
            Tensor: The {user x feature} starting factors.
        """
        history = interactions.get_sparse().tocsr().astype(np.float32)
        history.data[:] = 1.0

        taken = np.maximum(np.diff(history.indptr), 1)
        summed = history @ item_factors.numpy()
        return torch.from_numpy(summed / taken[:, None]).float()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive, negative = batch

        user_f = self.user_factors(user)
        positive_f = self.item_factors(positive)
        negative_f = self.item_factors(negative)
        positive_b = self.item_bias(positive).squeeze(-1)
        negative_b = self.item_bias(negative).squeeze(-1)

        positive_score = positive_b + (user_f * positive_f).sum(dim=-1)
        negative_score = negative_b + (user_f * negative_f).sum(dim=-1)

        loss = (
            self.rec_loss(positive_score, negative_score)
            + self.reg_weight * self.reg_loss(user_f, positive_f, negative_f)
            + self.bias_reg_weight * self.reg_loss(positive_b, negative_b)
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Score each given user against the item beside it.

        Args:
            user (Tensor): The user indices.
            item (Tensor): The item indices.

        Returns:
            Tensor: One score per pair.
        """
        return self.item_bias(item).squeeze(-1) + (
            self.user_factors(user) * self.item_factors(item)
        ).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the feature factors and the item biases.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_f = self.user_factors(user_indices)

        if item_indices is None:
            return (
                user_f @ self.item_factors.weight[:-1].t()
                + self.item_bias.weight[:-1].t()
            )

        return torch.einsum(
            "bf,bsf->bs", user_f, self.item_factors(item_indices)
        ) + self.item_bias(item_indices).squeeze(-1)
