# pylint: disable = R0801, E1102
from typing import Any, List, Optional

import torch
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class Tower(nn.Module):
    """One side of the model: an encoder from an index to a vector.

    Args:
        input_size (int): The width of what the tower is given.
        hidden_size (List[int]): The width of each hidden layer, the last of
            which is the vector the tower produces.
        dropout (float): The dropout applied between layers.
    """

    def __init__(self, input_size: int, hidden_size: List[int], dropout: float):
        super().__init__()

        layers: List[nn.Module] = []
        previous = input_size
        for index, width in enumerate(hidden_size):
            layers.append(nn.Linear(previous, width))
            # The last layer produces the vector that is scored, so it is left
            # linear: an activation there would bend the inner product space the
            # retrieval is defined in.
            if index < len(hidden_size) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            previous = width

        self.encoder = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Encode the given features into the retrieval space.

        Args:
            features (Tensor): What the tower is given, [..., input_size].

        Returns:
            Tensor: The encoded vectors, [..., hidden_size[-1]].
        """
        return self.encoder(features)


@model_registry.register(name="TwoTower")
class TwoTower(IterativeRecommender):
    """Implementation of the two-tower retrieval model from
        Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
        Recommendations (RecSys 2019)

    A user and an item are encoded independently and scored by the inner product
    of the two vectors. Because the item side never sees the user, the whole
    catalogue can be encoded once and searched, which is what the architecture is
    for; and because the item tower can be given the item's attributes, an item
    with no interactions still receives a vector.

    Training draws its negatives from the batch itself: every other item present
    is treated as a negative for a given user, which is what makes one pass over
    the positives enough. Items sampled that way appear in proportion to their
    popularity, so the logits are corrected for it, otherwise the model learns to
    push popular items down rather than to rank them.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        interactions (Optional[Interactions]): The training interactions, read for
            the item attributes and the item frequencies.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the user and item embeddings.
        tower_hidden_size (List[int]): The width of each tower layer, the last of
            which is the retrieval space.
        dropout (float): The dropout applied inside the towers.
        temperature (float): The scale applied to the logits before the softmax.
        use_item_features (bool): Whether the item tower is also given the item's
            attributes, which is what lets it place an unseen item.
        correct_sampling_bias (bool): Whether the in-batch logits are corrected
            for how often each item is drawn.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        item_features (Optional[Tensor]): The item attributes, aligned with the
            item embedding and carrying its padding row. Empty when unused.
        log_item_frequency (Tensor): The log frequency of each item, which is how
            often it is drawn as an in-batch negative.

    Raises:
        ValueError: If the item attributes are asked for but none were provided.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_DATALOADER

    # Model hyperparameters
    embedding_size: int
    tower_hidden_size: List[int]
    dropout: float
    temperature: float
    use_item_features: bool
    correct_sampling_bias: bool
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    item_features: Optional[Tensor]
    log_item_frequency: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        interactions: Optional[Interactions] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        features = None
        if self.use_item_features:
            side = interactions.get_side_sparse() if interactions is not None else None
            if side is None:
                raise ValueError(
                    "TwoTower was asked to use item features but the dataset "
                    "carries no side information. Configure 'reader.side' or set "
                    "'use_item_features' to false."
                )
            # The padding row keeps the lookup aligned with the item embedding,
            # which carries one too.
            dense = torch.as_tensor(side.toarray(), dtype=torch.float)
            features = torch.cat([dense, torch.zeros(1, dense.size(1))], dim=0)

        self.register_buffer(
            "item_features", features if features is not None else torch.empty(0)
        )
        item_input = self.embedding_size + (
            features.size(1) if features is not None else 0
        )

        self.user_tower = Tower(
            self.embedding_size, self.tower_hidden_size, self.dropout
        )
        self.item_tower = Tower(item_input, self.tower_hidden_size, self.dropout)

        # How often each item is drawn as an in-batch negative is how often it
        # appears at all, so its frequency is what the correction needs.
        counts = torch.zeros(self.n_items + 1)
        if interactions is not None:
            observed = torch.as_tensor(
                interactions.get_sparse().getnnz(axis=0), dtype=torch.float
            )
            counts[: observed.numel()] = observed
        frequency = counts / counts.sum().clamp(min=1.0)
        self.register_buffer("log_item_frequency", frequency.clamp(min=1e-12).log())

        self.apply(self._init_weights)
        self.reg_loss = EmbLoss()

    def encode_users(self, user: Tensor) -> Tensor:
        """Place the given users in the retrieval space.

        Args:
            user (Tensor): The user indices.

        Returns:
            Tensor: The user vectors.
        """
        return self.user_tower(self.user_embedding(user))

    def encode_items(self, item: Tensor) -> Tensor:
        """Place the given items in the retrieval space.

        Args:
            item (Tensor): The item indices.

        Returns:
            Tensor: The item vectors.
        """
        embedded = self.item_embedding(item)
        if not self.use_item_features:
            return self.item_tower(embedded)

        return self.item_tower(torch.cat([embedded, self.item_features[item]], dim=-1))

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_positive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, item = batch

        user_vectors = self.encode_users(user)
        item_vectors = self.encode_items(item)

        # Every item in the batch is a candidate for every user in it, so one
        # matrix holds the positive on the diagonal and the negatives elsewhere.
        logits = user_vectors @ item_vectors.t() / self.temperature

        if self.correct_sampling_bias:
            logits = logits - self.log_item_frequency[item].unsqueeze(0)

        # The same item can appear twice in a batch, and it is not a negative for
        # the row it is the positive of.
        repeated = item.unsqueeze(0) == item.unsqueeze(1)
        repeated.fill_diagonal_(False)
        logits = logits.masked_fill(repeated, -torch.inf)

        target = torch.arange(len(user), device=logits.device)
        loss = nn.functional.cross_entropy(logits, target)

        loss = loss + self.reg_weight * self.reg_loss(
            self.user_embedding(user), self.item_embedding(item)
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
        return (self.encode_users(user) * self.encode_items(item)).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction by inner product in the retrieval space.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_vectors = self.encode_users(user_indices)

        if item_indices is None:
            catalogue = torch.arange(self.n_items, device=user_vectors.device)
            return user_vectors @ self.encode_items(catalogue).t()

        return torch.einsum("be,bse->bs", user_vectors, self.encode_items(item_indices))
