import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

import warprec.recommenders  # noqa: F401  (populates the model registry)
from warprec.recommenders.base_recommender import (
    ContextRecommenderUtils,
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.serving.errors import ServingError
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry

Label = Union[int, str]


@dataclass(frozen=True)
class ServingPolicy:
    """How an endpoint answers, as set in its configuration.

    Attributes:
        default_k (int): The list length when a request does not set one.
        max_k (int): The longest list a request may ask for.
        mask_seen (bool): Whether items seen in training are left out.
        unknown_user (Literal["error", "popular"]): The answer for a user
            unseen in training.
    """

    default_k: int = 10
    max_k: int = 100
    mask_seen: bool = True
    unknown_user: Literal["error", "popular"] = "error"


@dataclass
class Query:
    """A request resolved to the model's internal indices, ready to be batched.

    Attributes:
        k (int): How many items to return.
        user (Optional[int]): The internal user index; None for an anonymous session.
        history (Optional[List[int]]): The session's internal item indices, oldest first.
        exclude (List[int]): Internal item indices never to return.
        fallback (bool): Whether popularity answers because the user is unknown.
    """

    k: int
    user: Optional[int] = None
    history: Optional[List[int]] = None
    exclude: List[int] = field(default_factory=list)
    fallback: bool = False


def _builtin(label: Any) -> Label:
    """A mapping label as a JSON-friendly Python value.

    Args:
        label (Any): A label from the dataset mapping, possibly a numpy scalar.

    Returns:
        Label: The same label as a plain int or str.
    """
    return label.item() if isinstance(label, np.generic) else label


class ServableModel:
    """A trained model wrapped with everything a request needs around it.

    Requests speak in the dataset's own user and item ids. This class maps them
    to the model's indices and back, feeds a sequential model its history, keeps
    items the user has already seen out of the list, and falls back on
    popularity for a user the model has never seen. It knows nothing about Ray,
    so all of it is tested without a cluster.

    Args:
        model (Recommender): The restored model, already on its device.
        seen (Optional[csr_matrix]): The binary training matrix, when the
            checkpoint carries it.
        histories (Optional[Dict[str, np.ndarray]]): The packed training
            histories of a sequential model.
        policy (ServingPolicy): How the endpoint answers.
        item_names (Optional[Dict[str, str]]): Item names by external id.
        warprec_version (Optional[str]): The WarpRec that wrote the checkpoint.

    Raises:
        ValueError: If the policy asks for a popularity fallback the checkpoint
            has no data for.
    """

    def __init__(
        self,
        model: Recommender,
        seen: Optional[csr_matrix] = None,
        histories: Optional[Dict[str, np.ndarray]] = None,
        policy: ServingPolicy = ServingPolicy(),
        item_names: Optional[Dict[str, str]] = None,
        warprec_version: Optional[str] = None,
    ):
        if policy.unknown_user == "popular" and seen is None:
            raise ValueError(
                "unknown_user: popular ranks by how often items were seen in "
                "training, and this checkpoint does not carry that. Save the "
                "model again with this version of WarpRec."
            )
        if policy.mask_seen and seen is None:
            logger.attention(
                f"The {model.name} checkpoint does not record what each user has "
                "seen, so already seen items cannot be left out of its answers."
            )

        self._model = model
        self._seen = seen
        self._histories = histories
        self._policy = policy
        self._warprec_version = warprec_version

        info = model.info
        self._users = {
            str(label): index for label, index in info["user_mapping"].items()
        }
        self._items = {
            str(label): index for label, index in info["item_mapping"].items()
        }
        self._labels: List[Label] = [None] * info["n_items"]  # type: ignore[list-item]
        for label, index in info["item_mapping"].items():
            self._labels[index] = _builtin(label)

        self._names = item_names or {}
        self._by_name = {
            name: self._items[label]
            for label, name in self._names.items()
            if label in self._items
        }

        self._popularity: Optional[Tensor] = None
        if seen is not None:
            counts = np.asarray(seen.sum(axis=0), dtype=np.float32).ravel()
            self._popularity = torch.from_numpy(counts).to(model.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        policy: ServingPolicy = ServingPolicy(),
        item_names: Optional[Dict[str, str]] = None,
    ) -> "ServableModel":
        """Load a checkpoint written by the train pipeline, ready to answer.

        Checkpoints are pickles and can run code when loaded, so only files
        from a trusted source should be served.

        Args:
            path (Union[str, Path]): The .pth file.
            device (str): The device to run the model on.
            policy (ServingPolicy): How the endpoint answers.
            item_names (Optional[Dict[str, str]]): Item names by external id.

        Returns:
            ServableModel: The model, ready to answer.

        Raises:
            ValueError: If the model is context-aware, which serving does not
                support yet.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
        model_class = model_registry.get_class(checkpoint["name"])
        if issubclass(model_class, ContextRecommenderUtils):
            raise ValueError(
                f"{checkpoint['name']} is context-aware: every request would need a "
                "context vector, which serving does not accept yet."
            )

        model = model_class.from_checkpoint(checkpoint=checkpoint)
        model.to(device)
        model.eval()

        payload = checkpoint.get("serving") or {}
        return cls(
            model,
            seen=payload.get("seen"),
            histories=payload.get("histories"),
            policy=policy,
            item_names=item_names,
            warprec_version=checkpoint.get("warprec_version"),
        )

    @property
    def n_users(self) -> int:
        """The number of users seen in training."""
        return self._model.info["n_users"]

    @property
    def n_items(self) -> int:
        """The number of items the model scores."""
        return self._model.info["n_items"]

    @property
    def is_sequential(self) -> bool:
        """Whether the model scores from a sequence of items."""
        return isinstance(self._model, SequentialRecommenderUtils)

    @property
    def needs_user(self) -> bool:
        """Whether a session alone is not enough and the user must be known."""
        return self.is_sequential and getattr(self._model, "needs_user", False)

    def describe(self) -> Dict[str, Any]:
        """What the endpoint serves, for clients to discover.

        Returns:
            Dict[str, Any]: The model name, kind, sizes and hyperparameters.
        """
        return {
            "model": self._model.name,
            "kind": "sequential" if self.is_sequential else "general",
            "n_users": self.n_users,
            "n_items": self.n_items,
            "needs_user": self.needs_user,
            "warprec_version": self._warprec_version,
            # Round-tripped through JSON so that whatever types the
            # hyperparameters carry reach the client as plain values.
            "params": json.loads(json.dumps(self._model.get_params(), default=str)),
        }

    def resolve(
        self,
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
        k: Optional[int] = None,
        exclude: Optional[List[Label]] = None,
    ) -> Query:
        """Check a request and translate it to the model's indices.

        Args:
            user_id (Optional[Label]): The user, by the dataset's own id.
            history (Optional[List[Label]]): A session of item ids or names,
                oldest first. Sequential models only.
            k (Optional[int]): How many items to return.
            exclude (Optional[List[Label]]): Items never to return.

        Returns:
            Query: The request in internal indices.

        Raises:
            ServingError: If the request cannot be answered, with the reason.
        """
        k = self._policy.default_k if k is None else k
        if not 1 <= k <= self._policy.max_k:
            raise ServingError(
                422, f"k must be between 1 and {self._policy.max_k}, got {k}."
            )
        k = min(k, self.n_items)
        excluded = [self._item_index(token) for token in exclude or []]
        user = None if user_id is None else self._users.get(str(user_id))

        if history is not None:
            return self._resolve_session(user, history, k, excluded)
        if user_id is None:
            raise ServingError(
                422, "Send a user_id, or a history when the model is sequential."
            )
        if user is None:
            if self._policy.unknown_user == "popular":
                return Query(k=k, exclude=excluded, fallback=True)
            raise ServingError(404, f"User '{user_id}' is not in the training data.")
        return Query(
            k=k, user=user, history=self._stored_history(user), exclude=excluded
        )

    # Scores come out of the model as inference tensors, and masking edits them
    # in place, so the whole path - scoring, masking, ranking - runs in the mode.
    @torch.inference_mode()
    def recommend(self, queries: Sequence[Query]) -> List[List[Dict[str, Any]]]:
        """Answer a batch of queries with one forward pass.

        Args:
            queries (Sequence[Query]): The resolved queries.

        Returns:
            List[List[Dict[str, Any]]]: For each query, its items, best first.
        """
        scored = [query for query in queries if not query.fallback]
        rows = iter(self._predict(scored)) if scored else iter(())
        answers = []
        for query in queries:
            if query.fallback:
                scores = self._popularity.clone()  # type: ignore[union-attr]
            else:
                scores = next(rows)
            answers.append(self._top_k(scores, query))
        return answers

    @torch.inference_mode()
    def score(
        self,
        items: List[Label],
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
    ) -> List[Dict[str, Any]]:
        """The model's score for each candidate, in the order they were sent.

        Nothing is masked: a caller re-ranking its own candidates asked about
        exactly these items.

        Args:
            items (List[Label]): The candidates, by id or name.
            user_id (Optional[Label]): The user, by the dataset's own id.
            history (Optional[List[Label]]): A session, for a sequential model.

        Returns:
            List[Dict[str, Any]]: One entry per candidate, in request order.

        Raises:
            ServingError: If there are no candidates, or the request cannot be answered.
        """
        if not items:
            raise ServingError(422, "items must hold at least one candidate.")
        candidates = [self._item_index(token) for token in items]
        query = self.resolve(user_id=user_id, history=history, k=1)
        if query.fallback:
            row = self._popularity
        else:
            row = self._predict([query])[0]
        values = row[torch.tensor(candidates, device=row.device)].tolist()  # type: ignore[index]
        return [self._entry(index, value) for index, value in zip(candidates, values)]

    def _resolve_session(
        self, user: Optional[int], history: List[Label], k: int, excluded: List[int]
    ) -> Query:
        """Check a request that sends its own history.

        Args:
            user (Optional[int]): The internal user index, if the user is known.
            history (List[Label]): The session's item ids or names.
            k (int): How many items to return.
            excluded (List[int]): Internal item indices never to return.

        Returns:
            Query: The request in internal indices.

        Raises:
            ServingError: If the model cannot score a session like this one.
        """
        if not self.is_sequential:
            raise ServingError(
                422,
                f"{self._model.name} is not sequential: send a user_id, not a history.",
            )
        if not history:
            raise ServingError(422, "history must hold at least one item.")
        if user is None and self.needs_user:
            raise ServingError(
                422,
                f"{self._model.name} mixes the user into the sequence, so a history "
                "needs a known user_id alongside it.",
            )
        items = [self._item_index(token) for token in history]
        return Query(k=k, user=user, history=items, exclude=excluded)

    def _stored_history(self, user: int) -> Optional[List[int]]:
        """The training history of a known user, for a sequential model.

        Args:
            user (int): The internal user index.

        Returns:
            Optional[List[int]]: The user's most recent items, or None for a
                model that does not read sequences.

        Raises:
            ServingError: If the checkpoint stores no histories to read from.
        """
        if not self.is_sequential:
            return None
        if self._histories is None:
            raise ServingError(
                422,
                "This checkpoint stores no training histories: send the user's "
                "history with the request.",
            )
        offsets = self._histories["offsets"]
        return self._histories["items"][offsets[user] : offsets[user + 1]].tolist()

    def _item_index(self, token: Label) -> int:
        """An item's internal index from its id or, with a catalogue, its name.

        Args:
            token (Label): The item id or name.

        Returns:
            int: The internal index.

        Raises:
            ServingError: If the item is unknown.
        """
        index = self._items.get(str(token))
        if index is None:
            index = self._by_name.get(str(token))
        if index is None:
            raise ServingError(422, f"Item '{token}' is not in the model's catalogue.")
        return index

    def _predict(self, queries: Sequence[Query]) -> Tensor:
        """Score every item for a batch of queries in one call.

        Args:
            queries (Sequence[Query]): Queries answered by the model.

        Returns:
            Tensor: The scores, one row per query.
        """
        device = self._model.device
        # An anonymous session has no user; models that read sequences only
        # ignore the index, and those that do not were refused in resolve().
        users = [query.user if query.user is not None else 0 for query in queries]
        inputs: Dict[str, Any] = {
            "user_indices": torch.tensor(users, dtype=torch.long, device=device)
        }
        if self.is_sequential:
            # Every row is right-padded to the model's full width, not to the
            # longest row of the batch: attention models build their causal mask
            # for max_seq_len positions and cannot read a narrower batch.
            longest = self._model.max_seq_len
            recent = [query.history[-longest:] for query in queries]
            sequences = torch.full(
                (len(queries), longest), self.n_items, dtype=torch.long
            )
            for row, items in enumerate(recent):
                sequences[row, : len(items)] = torch.tensor(items, dtype=torch.long)
            inputs["user_seq"] = sequences.to(device)
            inputs["seq_len"] = torch.tensor(
                [len(items) for items in recent], dtype=torch.long, device=device
            )
        with torch.inference_mode():
            return self._model.predict(**inputs).float()

    def _top_k(self, scores: Tensor, query: Query) -> List[Dict[str, Any]]:
        """The best items of one row of scores, hidden ones left out.

        Args:
            scores (Tensor): The row of scores, modified in place.
            query (Query): The query it answers.

        Returns:
            List[Dict[str, Any]]: The items, best first. Fewer than k when
                masking leaves fewer items to rank.
        """
        hidden = self._hidden_items(query)
        if hidden:
            scores[torch.tensor(hidden, device=scores.device)] = -math.inf
        values, indices = torch.topk(scores, query.k)
        keep = torch.isfinite(values)
        return [
            self._entry(index, score)
            for index, score in zip(indices[keep].tolist(), values[keep].tolist())
        ]

    def _hidden_items(self, query: Query) -> List[int]:
        """The items a query must not get back.

        Args:
            query (Query): The query.

        Returns:
            List[int]: Internal indices of excluded and, with masking on, seen items.
        """
        hidden = list(query.exclude)
        if self._policy.mask_seen:
            if query.user is not None and self._seen is not None:
                start, end = (
                    self._seen.indptr[query.user],
                    self._seen.indptr[query.user + 1],
                )
                hidden.extend(self._seen.indices[start:end].tolist())
            if query.history:
                hidden.extend(query.history)
        return hidden

    def _entry(self, index: int, score: float) -> Dict[str, Any]:
        """One item of an answer.

        Args:
            index (int): The internal item index.
            score (float): Its score.

        Returns:
            Dict[str, Any]: The external id, the score and, with a catalogue, the name.
        """
        label = self._labels[index]
        entry: Dict[str, Any] = {"item_id": label, "score": score}
        if self._names:
            entry["name"] = self._names.get(str(label))
        return entry
