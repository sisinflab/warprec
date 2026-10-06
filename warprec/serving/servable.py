import difflib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

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
from warprec.serving.catalogue import Catalogue
from warprec.serving.context import ContextSchema
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


@dataclass(frozen=True)
class Presentation:
    """How an endpoint speaks about itself, as set in its configuration.

    Attributes:
        description (Optional[str]): What the endpoint serves, in words.
        item_noun (str): What an item is called, such as 'movie'.
        context_descriptions (Dict[str, str]): What each context field means.
    """

    description: Optional[str] = None
    item_noun: str = "item"
    context_descriptions: Dict[str, str] = field(default_factory=dict)


@dataclass
class Query:
    """A request resolved to the model's internal indices, ready to be batched.

    Attributes:
        k (int): How many items to return.
        user (Optional[int]): The internal user index; None for an anonymous session.
        history (Optional[List[int]]): The session's internal item indices, oldest first.
        exclude (List[int]): Internal item indices never to return.
        fallback (bool): Whether popularity answers because the user is unknown.
        context (Optional[List[Any]]): The encoded context row, for a
            context-aware model.
        candidates (List[int]): Internal indices of the items to score, for a
            scoring request.
        allowed (Optional[np.ndarray]): Internal indices of the only items a
            filtered request may return; None when it is not filtered.
        explain (bool): Whether each item comes with the training evidence
            linking it to the request's own items.
    """

    k: int
    user: Optional[int] = None
    history: Optional[List[int]] = None
    exclude: List[int] = field(default_factory=list)
    fallback: bool = False
    context: Optional[List[Any]] = None
    candidates: List[int] = field(default_factory=list)
    allowed: Optional[np.ndarray] = None
    explain: bool = False


def scores_on_device(path: Union[str, Path]) -> bool:
    """Whether the model in a checkpoint can do its scoring on a GPU.

    Closed-form models - the neighbourhood and EASE families, SLIM and the like
    - keep what they learned in numpy arrays rather than tensors, so they score
    on the CPU wherever the model is placed. Their state dict is empty, which
    is what tells them apart. Tensors are memory-mapped rather than read, so
    the check stays cheap for large models.

    Args:
        path (Union[str, Path]): The .pth file.

    Returns:
        bool: True when the model holds parameters or buffers it scores with.
    """
    checkpoint = torch.load(  # nosec B614
        path, map_location="cpu", weights_only=False, mmap=True
    )
    return len(checkpoint["state_dict"]) > 0


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
        payload (Optional[Dict[str, Any]]): What the checkpoint carries for
            serving: the training matrix ('seen'), the packed histories of a
            sequential model ('histories'), the context vocabulary and its
            counts ('context_maps', 'context_stats') and the training facts
            ('training'). Older checkpoints carry less, or nothing.
        policy (ServingPolicy): How the endpoint answers.
        catalogue (Optional[Catalogue]): The names and attributes of the items.
        presentation (Presentation): How the endpoint speaks about itself.
        warprec_version (Optional[str]): The WarpRec that wrote the checkpoint.

    Raises:
        ValueError: If the policy asks for a popularity fallback the checkpoint
            has no data for, or the model is context-aware and the checkpoint
            lacks its context vocabulary.
    """

    def __init__(
        self,
        model: Recommender,
        payload: Optional[Dict[str, Any]] = None,
        policy: ServingPolicy = ServingPolicy(),
        catalogue: Optional[Catalogue] = None,
        presentation: Presentation = Presentation(),
        warprec_version: Optional[str] = None,
    ):
        payload = payload or {}
        seen: Optional[csr_matrix] = payload.get("seen")
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
        self._histories: Optional[Dict[str, np.ndarray]] = payload.get("histories")
        self._policy = policy
        self._presentation = presentation
        self._training: Optional[Dict[str, Any]] = payload.get("training")
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

        self._catalogue = catalogue

        self._popularity: Optional[Tensor] = None
        if seen is not None:
            counts = np.asarray(seen.sum(axis=0), dtype=np.float32).ravel()
            self._popularity = torch.from_numpy(counts).to(model.device)

        # A context-aware model needs the vocabulary its contexts were encoded
        # with, which only checkpoints saved by this version carry.
        self._context: Optional[ContextSchema] = None
        if isinstance(model, ContextRecommenderUtils) and model.context_labels:
            if payload.get("context_maps") is None:
                raise ValueError(
                    f"{model.name} is context-aware, but this checkpoint does not "
                    "carry the context values it was trained on. Save the model "
                    "again with this version of WarpRec."
                )
            self._context = ContextSchema.from_info(
                model.info,
                payload["context_maps"],
                stats=payload.get("context_stats"),
                descriptions=presentation.context_descriptions,
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        policy: ServingPolicy = ServingPolicy(),
        catalogue: Optional[Catalogue] = None,
        presentation: Presentation = Presentation(),
    ) -> "ServableModel":
        """Load a checkpoint written by the train pipeline, ready to answer.

        Checkpoints are pickles and can run code when loaded, so only files
        from a trusted source should be served.

        Args:
            path (Union[str, Path]): The .pth file.
            device (str): The device to run the model on.
            policy (ServingPolicy): How the endpoint answers.
            catalogue (Optional[Catalogue]): The names and attributes of the items.
            presentation (Presentation): How the endpoint speaks about itself.

        Returns:
            ServableModel: The model, ready to answer.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
        model_class = model_registry.get_class(checkpoint["name"])

        model = model_class.from_checkpoint(checkpoint=checkpoint)
        model.to(device)
        model.eval()

        return cls(
            model,
            payload=checkpoint.get("serving"),
            policy=policy,
            catalogue=catalogue,
            presentation=presentation,
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

    @property
    def context_schema(self) -> Optional[ContextSchema]:
        """The context fields of a context-aware model; None for any other."""
        return self._context

    def describe(self) -> Dict[str, Any]:
        """The model card: what the endpoint serves and how to ask it.

        It is what a client or an agent needs before a first request: the kind
        of model, what a request must contain with an example that works as
        is, the items it knows, what it was trained on and how well it scored,
        and the context it accepts.

        Returns:
            Dict[str, Any]: The card.
        """
        context = self.describe_context() if self._context is not None else None
        return {
            "model": self._model.name,
            "kind": "sequential" if self.is_sequential else "general",
            "description": self._presentation.description,
            "item_noun": self._presentation.item_noun,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "needs_user": self.needs_user,
            "how_to_ask": self._how_to_ask(),
            "example_request": self._example_request(),
            "catalogue": self._catalogue_card(),
            "training": self._training,
            "context": context["fields"] if context else None,
            "example_context": context["example"] if context else None,
            "warprec_version": self._warprec_version,
            # Round-tripped through JSON so that whatever types the
            # hyperparameters carry reach the client as plain values.
            "params": json.loads(json.dumps(self._model.get_params(), default=str)),
        }

    def describe_context(self) -> Dict[str, Any]:
        """The context a context-aware model accepts, with an example.

        Returns:
            Dict[str, Any]: Each field's type, accepted values (most frequent
                in training first, with their counts when known), range for a
                numeric field and description; and an example context.

        Raises:
            ServingError: If the model does not use context.
        """
        if self._context is None:
            raise ServingError(
                422, f"{self._model.name} does not use context: there is none to give."
            )
        return {"fields": self._context.describe(), "example": self._context.example()}

    def _how_to_ask(self) -> List[str]:
        """What a request must and may contain, in sentences.

        Returns:
            List[str]: The instructions.
        """
        nouns = f"{self._presentation.item_noun}s"
        by = "id or exact name" if self._catalogue is not None else "id"
        sentences = [f"Send user_id: one of the {self.n_users} users seen in training."]
        if self._policy.unknown_user == "popular":
            sentences.append(f"An unknown user gets the most popular {nouns}.")
        else:
            sentences.append("An unknown user is refused.")
        if self.is_sequential:
            together = " together with the user_id" if self.needs_user else ""
            sentences.append(
                f"Or send history{together}: the {nouns} of a session, oldest "
                f"first, by {by}."
            )
        if self._context is not None:
            sentences.append(
                "Every request needs context: a value for each of "
                f"{', '.join(self._context.labels)} (see 'context' for the "
                "accepted values)."
            )
        sentences.append(
            f"Optional: k, how many {nouns} to return (1 to {self._policy.max_k}, "
            f"default {self._policy.default_k}), and exclude, {nouns} to leave out."
        )
        attributes = self._catalogue.attribute_names() if self._catalogue else []
        if attributes:
            sentences.append(
                f"Optional: filter, to keep only {nouns} with given attributes "
                f"({', '.join(attributes)}), such as "
                f'{{"{attributes[0]}": "<value>"}}; a list matches any of its values.'
            )
        sentences.append(
            f"Optional: explain: true adds to each {self._presentation.item_noun} "
            f"the {nouns} of the request that training users most often consumed "
            "with it - evidence from the data, not the model's reasoning."
        )
        return sentences

    def _example_request(self) -> Dict[str, Any]:
        """A request this endpoint answers, to start from.

        Returns:
            Dict[str, Any]: The request fields.
        """
        request: Dict[str, Any] = {"k": min(5, self._policy.max_k)}
        if self.is_sequential and not self.needs_user:
            # A session of the best-known items reads naturally as an example.
            popular = self._most_interacted(3)
            request["history"] = [
                (
                    self._catalogue.names.get(str(self._labels[index]))
                    if self._catalogue
                    else None
                )
                or self._labels[index]
                for index in popular
            ]
        else:
            request["user_id"] = self._example_user()
        if self._context is not None:
            request["context"] = self._context.example()
        return request

    def _example_user(self) -> Label:
        """A user the model learned from, for the example request.

        Returns:
            Label: The id of the first user with training interactions, or of
                the first user when the checkpoint does not record them.
        """
        mapping = self._model.info["user_mapping"]
        if self._seen is not None:
            active = np.flatnonzero(np.diff(self._seen.indptr))
            if active.size:
                first = int(active[0])
                for label, index in mapping.items():
                    if index == first:
                        return _builtin(label)
        return _builtin(next(iter(mapping)))

    def _catalogue_card(self) -> Dict[str, Any]:
        """What the endpoint knows about its items.

        Returns:
            Dict[str, Any]: Whether items have names, how many there are, and
                for each attribute its most common values.
        """
        if self._catalogue is None:
            return {"names": False, "n_items": self.n_items, "attributes": {}}
        attributes = {}
        for attribute in self._catalogue.attribute_names():
            counts = self._catalogue.attribute_values(attribute)
            attributes[attribute] = {
                "examples": [value for value, _ in counts.most_common(10)],
                "n_values": len(counts),
            }
        return {"names": True, "n_items": self.n_items, "attributes": attributes}

    def _most_interacted(self, k: int) -> List[int]:
        """The internal indices of the items with the most training interactions.

        Args:
            k (int): How many.

        Returns:
            List[int]: The indices, most interacted first; the first items when
                the checkpoint does not record interactions.
        """
        if self._popularity is None:
            return list(range(min(k, self.n_items)))
        return torch.topk(self._popularity, min(k, self.n_items)).indices.tolist()

    def resolve(
        self,
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
        k: Optional[int] = None,
        exclude: Optional[List[Label]] = None,
        context: Optional[Union[Dict[str, Any], List[Any]]] = None,
        filter: Optional[Dict[str, Any]] = None,  # pylint: disable=redefined-builtin
        explain: bool = False,
    ) -> Query:
        """Check a request and translate it to the model's indices.

        Args:
            user_id (Optional[Label]): The user, by the dataset's own id.
            history (Optional[List[Label]]): A session of item ids or names,
                oldest first. Sequential models only.
            k (Optional[int]): How many items to return.
            exclude (Optional[List[Label]]): Items never to return.
            context (Optional[Union[Dict[str, Any], List[Any]]]): The situation
                of the request. Required by context-aware models, refused by others.
            filter (Optional[Dict[str, Any]]): Item attributes the answer must
                have, such as {"genres": "Comedy"}; a list matches any of its values.
            explain (bool): Whether each item comes with the training evidence
                linking it to the request's own items.

        Returns:
            Query: The request in internal indices.

        Raises:
            ServingError: If the request cannot be answered, with the reason.
        """
        if self._context is None:
            if context is not None:
                raise ServingError(
                    422, f"{self._model.name} does not use context: leave it out."
                )
            encoded = None
        else:
            if context is None:
                raise ServingError(
                    422,
                    f"{self._model.name} is context-aware: send a context with the "
                    f"fields {self._context.labels}.",
                )
            encoded = self._context.encode(context)

        query = self._resolve_target(user_id, history, k, exclude)
        query.context = encoded
        query.allowed = self._allowed_items(filter) if filter else None
        query.explain = explain
        return query

    def _resolve_target(
        self,
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
        k: Optional[int] = None,
        exclude: Optional[List[Label]] = None,
    ) -> Query:
        """Resolve the user, history, length and exclusions of a request.

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
        k = self._checked_k(k)
        excluded = [self._item_index(token) for token in exclude or []]
        user = None if user_id is None else self._known_user(user_id)

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
            answer = self._top_k(scores, query)
            if query.explain:
                self._explain(query, answer)
            answers.append(answer)
        return answers

    @torch.inference_mode()
    def popular_items(
        self,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,  # pylint: disable=redefined-builtin
        exclude: Optional[List[Label]] = None,
    ) -> List[Dict[str, Any]]:
        """The items with the most training interactions, for anyone.

        Args:
            k (Optional[int]): How many items to return.
            filter (Optional[Dict[str, Any]]): Item attributes the answer must have.
            exclude (Optional[List[Label]]): Items never to return.

        Returns:
            List[Dict[str, Any]]: The items, most interacted first, each with
                its number of training interactions.

        Raises:
            ServingError: If the checkpoint does not record training
                interactions, or the request is invalid.
        """
        if self._popularity is None:
            raise ServingError(
                422,
                "This checkpoint does not record training interactions, so there "
                "is no popularity to rank by. Save the model again with this "
                "version of WarpRec.",
            )
        query = Query(k=self._checked_k(k))
        query.exclude = [self._item_index(token) for token in exclude or []]
        query.allowed = self._allowed_items(filter) if filter else None
        answer = self._top_k(self._popularity.clone(), query, mask_seen=False)
        for entry in answer:
            entry["interactions"] = int(entry["score"])
        return answer

    def score(
        self,
        items: List[Label],
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
        context: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """The model's score for each candidate, in the order they were sent.

        Nothing is masked: a caller re-ranking its own candidates asked about
        exactly these items.

        Args:
            items (List[Label]): The candidates, by id or name.
            user_id (Optional[Label]): The user, by the dataset's own id.
            history (Optional[List[Label]]): A session, for a sequential model.
            context (Optional[Union[Dict[str, Any], List[Any]]]): The situation
                of the request, for a context-aware model.

        Returns:
            List[Dict[str, Any]]: One entry per candidate, in request order.
        """
        query = self.resolve_scoring(
            items, user_id=user_id, history=history, context=context
        )
        return self.score_batch([query])[0]

    def resolve_scoring(
        self,
        items: List[Label],
        user_id: Optional[Label] = None,
        history: Optional[List[Label]] = None,
        context: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> Query:
        """Check a scoring request and translate it to the model's indices.

        Args:
            items (List[Label]): The candidates, by id or name.
            user_id (Optional[Label]): The user, by the dataset's own id.
            history (Optional[List[Label]]): A session, for a sequential model.
            context (Optional[Union[Dict[str, Any], List[Any]]]): The situation
                of the request, for a context-aware model.

        Returns:
            Query: The request in internal indices, its candidates included.

        Raises:
            ServingError: If there are no candidates, or the request cannot be answered.
        """
        if not items:
            raise ServingError(422, "items must hold at least one candidate.")
        candidates = [self._item_index(token) for token in items]
        query = self.resolve(user_id=user_id, history=history, k=1, context=context)
        query.candidates = candidates
        return query

    # Like recommend: one forward pass for the batch, edited in inference mode.
    @torch.inference_mode()
    def score_batch(self, queries: Sequence[Query]) -> List[List[Dict[str, Any]]]:
        """Score the candidates of a batch of queries with one forward pass.

        Args:
            queries (Sequence[Query]): Queries returned by resolve_scoring().

        Returns:
            List[List[Dict[str, Any]]]: For each query, one entry per candidate,
                in the order they were sent.
        """
        scored = [query for query in queries if not query.fallback]
        rows = iter(self._predict(scored)) if scored else iter(())
        answers = []
        for query in queries:
            row = self._popularity if query.fallback else next(rows)
            values = row[  # type: ignore[index]
                torch.tensor(query.candidates, device=row.device)  # type: ignore[union-attr]
            ].tolist()
            answers.append(
                [
                    self._entry(index, value)
                    for index, value in zip(query.candidates, values)
                ]
            )
        return answers

    def search_items(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Find the items whose name contains some text.

        Answers "do you know this item?": an item that is not in the catalogue
        comes back as no match, with the closest names as suggestions.

        Args:
            query (str): Part of a name, matched without regard to case.
            limit (int): The most matches to return.

        Returns:
            Dict[str, Any]: The query, the matching items and, when nothing
                matched, the items with the closest names.

        Raises:
            ServingError: If the endpoint has no catalogue to search, or the
                limit is out of range.
        """
        catalogue = self._require_catalogue()
        if not 1 <= limit <= self._policy.max_k:
            raise ServingError(
                422, f"limit must be between 1 and {self._policy.max_k}, got {limit}."
            )
        matches = [
            self._items[item] for item in catalogue.search(query) if item in self._items
        ]
        # Past an exact match, the more interactions an item had, the likelier
        # it is the one meant.
        exact = matches[:1] if catalogue.find(query) is not None else []
        rest = sorted(
            (index for index in matches if index not in exact),
            key=self._interactions,
            reverse=True,
        )
        found = [self._catalogue_entry(index) for index in (exact + rest)[:limit]]
        suggestions = []
        if not found:
            for name in catalogue.suggest(query):
                item = catalogue.find(name)
                if item in self._items:
                    suggestions.append(self._catalogue_entry(self._items[item]))
        return {"query": query, "matches": found, "suggestions": suggestions}

    def get_items(self, items: List[Label]) -> Dict[str, Any]:
        """Look items up by id or name.

        Args:
            items (List[Label]): The ids or names.

        Returns:
            Dict[str, Any]: The items found, in request order, and for each one
                not found, what was asked and the closest names.
        """
        found, unknown = [], []
        for token in items:
            try:
                found.append(self._catalogue_entry(self._item_index(token)))
            except ServingError:
                close = self._catalogue.suggest(str(token)) if self._catalogue else []
                unknown.append({"query": token, "suggestions": close})
        return {"items": found, "unknown": unknown}

    def _require_catalogue(self) -> Catalogue:
        """The endpoint's catalogue, for operations that need item names.

        Returns:
            Catalogue: The catalogue.

        Raises:
            ServingError: If the endpoint has none.
        """
        if self._catalogue is None:
            raise ServingError(
                422,
                "This endpoint has no item catalogue: items are known by id only. "
                "Configure item_metadata to search them by name.",
            )
        return self._catalogue

    def _interactions(self, index: int) -> int:
        """How many training interactions an item had.

        Args:
            index (int): The internal item index.

        Returns:
            int: The count; 0 when the checkpoint does not carry it.
        """
        return 0 if self._popularity is None else int(self._popularity[index].item())

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

    def _known_user(self, user_id: Label) -> Optional[int]:
        """The internal index of a user the model actually learned.

        A cold-start protocol keeps the users it holds out in the mapping with
        no training interactions, so their embedding never trained. With the
        training matrix at hand, such a user counts as unknown.

        Args:
            user_id (Label): The user, by the dataset's own id.

        Returns:
            Optional[int]: The internal index, or None for an unknown user.
        """
        user = self._users.get(str(user_id))
        if user is not None and self._seen is not None:
            if self._seen.indptr[user] == self._seen.indptr[user + 1]:
                return None
        return user

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
        if index is None and self._catalogue is not None:
            index = self._items.get(self._catalogue.find(str(token)) or "")
        if index is None:
            message = f"Item '{token}' is not in the model's catalogue."
            if self._catalogue is not None:
                close = self._catalogue.suggest(str(token))
                if close:
                    message += (
                        f" Did you mean {', '.join(repr(name) for name in close)}?"
                    )
            raise ServingError(422, message)
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
            longest = cast(SequentialRecommenderUtils, self._model).max_seq_len
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
        if self._context is not None:
            inputs["contexts"] = self._context.tensor(
                [query.context for query in queries], device
            )
        with torch.inference_mode():
            return self._model.predict(**inputs).float()

    def _top_k(
        self, scores: Tensor, query: Query, mask_seen: bool = True
    ) -> List[Dict[str, Any]]:
        """The best items of one row of scores, hidden ones left out.

        Args:
            scores (Tensor): The row of scores, modified in place.
            query (Query): The query it answers.
            mask_seen (bool): Whether the items the request has already seen
                are hidden, as the endpoint's policy says; off for popularity,
                which is the same for everyone.

        Returns:
            List[Dict[str, Any]]: The items, best first. Fewer than k when
                masking leaves fewer items to rank.
        """
        hidden = self._hidden_items(query) if mask_seen else list(query.exclude)
        if hidden:
            scores[torch.tensor(hidden, device=scores.device)] = -math.inf
        if query.allowed is not None:
            blocked = torch.ones_like(scores, dtype=torch.bool)
            blocked[torch.as_tensor(query.allowed, device=scores.device)] = False
            scores[blocked] = -math.inf
        values, indices = torch.topk(scores, query.k)
        keep = torch.isfinite(values)
        return [
            self._entry(index, score)
            for index, score in zip(indices[keep].tolist(), values[keep].tolist())
        ]

    def _allowed_items(self, filter: Dict[str, Any]) -> np.ndarray:  # pylint: disable=redefined-builtin
        """The items whose attributes match a filter.

        Values are compared without regard to case. A list of values matches
        an item that has any of them, and an attribute holding a list matches
        when any of its values does; several attributes must all match.

        Args:
            filter (Dict[str, Any]): Attribute values, such as {"genres": "Comedy"}.

        Returns:
            np.ndarray: The internal indices of the matching items.

        Raises:
            ServingError: If there is no catalogue, or an attribute or a value
                is unknown.
        """
        catalogue = self._require_catalogue()
        known = catalogue.attribute_names()
        allowed: Optional[set] = None
        for attribute, wanted in filter.items():
            if attribute not in known:
                raise ServingError(
                    422,
                    f"Unknown attribute '{attribute}'. Items can be filtered on: "
                    f"{known or 'nothing (no attributes are configured)'}.",
                )
            values = {str(value) for value in catalogue.attribute_values(attribute)}
            folded = {value.casefold(): value for value in values}
            requested = wanted if isinstance(wanted, list) else [wanted]
            for value in requested:
                if str(value).casefold() not in folded:
                    close = difflib.get_close_matches(str(value), list(values), n=3)
                    hint = (
                        f" Did you mean {', '.join(map(repr, close))}?" if close else ""
                    )
                    raise ServingError(422, f"No item has {attribute} '{value}'.{hint}")
            targets = {str(value).casefold() for value in requested}
            matching = set()
            for item, attributes in catalogue.attributes.items():
                value = attributes.get(attribute)
                cells = value if isinstance(value, list) else [value]
                if item in self._items and targets & {str(c).casefold() for c in cells}:
                    matching.add(self._items[item])
            allowed = matching if allowed is None else allowed & matching
        return np.array(sorted(allowed or set()), dtype=np.int64)

    def _explain(self, query: Query, answer: List[Dict[str, Any]]) -> None:
        """Add to each item the request's own items most often consumed with it.

        This is evidence from the training data - how many users interacted
        with both items - not the model's reasoning, which it does not expose.

        Args:
            query (Query): The query the answer is for.
            answer (List[Dict[str, Any]]): The items, completed in place.
        """
        own = set(query.history or [])
        if query.user is not None and self._seen is not None:
            start, end = (
                self._seen.indptr[query.user],
                self._seen.indptr[query.user + 1],
            )
            own.update(self._seen.indices[start:end].tolist())
        if not answer or not own or self._seen is None:
            for entry in answer:
                entry["because"] = []
            return
        own_items = sorted(own)
        targets = [self._items[str(entry["item_id"])] for entry in answer]
        columns = self._seen.tocsc()
        together = (columns[:, own_items].T @ columns[:, targets]).toarray()
        for position, entry in enumerate(answer):
            counts = together[:, position]
            best = [
                row for row in np.argsort(-counts, kind="stable")[:2] if counts[row] > 0
            ]
            entry["because"] = [
                {
                    "item_id": self._labels[own_items[row]],
                    **self._described(own_items[row]),
                    "co_occurrences": int(counts[row]),
                }
                for row in best
            ]

    def _checked_k(self, k: Optional[int]) -> int:
        """A requested list length, checked against the endpoint's limits.

        Args:
            k (Optional[int]): The requested length; None for the default.

        Returns:
            int: The length to return, at most the number of items.

        Raises:
            ServingError: If it is out of range.
        """
        k = self._policy.default_k if k is None else k
        if not 1 <= k <= self._policy.max_k:
            raise ServingError(
                422, f"k must be between 1 and {self._policy.max_k}, got {k}."
            )
        return min(k, self.n_items)

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
            Dict[str, Any]: The external id, the score and, with a catalogue,
                the name and attributes.
        """
        entry: Dict[str, Any] = {"item_id": self._labels[index], "score": score}
        entry.update(self._described(index))
        return entry

    def _described(self, index: int) -> Dict[str, Any]:
        """What the catalogue says about an item.

        Args:
            index (int): The internal item index.

        Returns:
            Dict[str, Any]: The name and the attributes, or nothing without a
                catalogue.
        """
        if self._catalogue is None:
            return {}
        label = str(self._labels[index])
        described: Dict[str, Any] = {"name": self._catalogue.names.get(label)}
        attributes = self._catalogue.attributes.get(label)
        if attributes:
            described["attributes"] = attributes
        return described

    def _catalogue_entry(self, index: int) -> Dict[str, Any]:
        """An item as search and lookup return it.

        Args:
            index (int): The internal item index.

        Returns:
            Dict[str, Any]: The external id, what the catalogue says about it,
                and how many training interactions it had.
        """
        entry: Dict[str, Any] = {"item_id": self._labels[index]}
        entry.update(self._described(index))
        entry["interactions"] = (
            self._interactions(index) if self._popularity is not None else None
        )
        return entry
