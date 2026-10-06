from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Label = Union[int, str]


class RecommendRequest(BaseModel):
    """A request for recommendations.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown fields are rejected.
        user_id (Optional[Label]): The user, by the dataset's own id.
        history (Optional[List[Label]]): A session of item ids or names, oldest
            first. Sequential models only.
        k (Optional[int]): How many items to return.
        exclude (List[Label]): Items never to return.
        context (Optional[Union[Dict[str, Any], List[Any]]]): The situation of
            the request, for a context-aware model: each field's value by name,
            or a list in the order the model describes.
        filter (Optional[Dict[str, Union[str, List[str]]]]): Item attributes
            the answer must have, such as {"genres": "Comedy"}; a list matches
            any of its values.
        explain (bool): Whether each item comes with the request's own items
            that training users most often consumed with it.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[Label] = None
    history: Optional[List[Label]] = None
    k: Optional[int] = None
    exclude: List[Label] = Field(default_factory=list)
    context: Optional[Union[Dict[str, Any], List[Any]]] = None
    filter: Optional[Dict[str, Union[str, List[str]]]] = None
    explain: bool = False


class ScoreRequest(BaseModel):
    """A request to score given candidates.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown fields are rejected.
        user_id (Optional[Label]): The user, by the dataset's own id.
        history (Optional[List[Label]]): A session, for a sequential model.
        items (List[Label]): The candidates, by id or name.
        context (Optional[Union[Dict[str, Any], List[Any]]]): The situation of
            the request, for a context-aware model.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[Label] = None
    history: Optional[List[Label]] = None
    items: List[Label] = Field(max_length=10_000)
    context: Optional[Union[Dict[str, Any], List[Any]]] = None


class ItemScore(BaseModel):
    """One item of an answer.

    Attributes:
        item_id (Label): The dataset's id of the item.
        score (float): The model's score; for popular items, the number of
            training interactions.
        name (Optional[str]): The item's name, when a catalogue is configured.
        attributes (Optional[Dict[str, Any]]): The item's attributes, when the
            catalogue has some.
        interactions (Optional[int]): The item's training interactions, for
            popular items.
        because (Optional[List[Dict[str, Any]]]): With explain, the request's
            own items that training users most often consumed with this one,
            and how often. Evidence from the data, not the model's reasoning.
    """

    item_id: Label
    score: float
    name: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    interactions: Optional[int] = None
    because: Optional[List[Dict[str, Any]]] = None


class RecommendResponse(BaseModel):
    """Recommendations, best first.

    Attributes:
        model (str): The endpoint that answered.
        items (List[ItemScore]): The items.
        fallback (bool): Whether popularity answered for an unknown user.
    """

    model: str
    items: List[ItemScore]
    fallback: bool


class ScoreResponse(BaseModel):
    """Scores of the candidates, in request order.

    Attributes:
        model (str): The endpoint that answered.
        scores (List[ItemScore]): One entry per candidate.
    """

    model: str
    scores: List[ItemScore]


class CatalogueItem(BaseModel):
    """An item as search and lookup return it.

    Attributes:
        item_id (Label): The dataset's id of the item.
        name (Optional[str]): Its name.
        attributes (Optional[Dict[str, Any]]): Its attributes.
        interactions (Optional[int]): How many training interactions it had.
    """

    item_id: Label
    name: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    interactions: Optional[int] = None


class SearchResponse(BaseModel):
    """The items whose name contains a query.

    Attributes:
        model (str): The endpoint that answered.
        query (str): What was searched.
        matches (List[CatalogueItem]): The matching items, best first.
        suggestions (List[CatalogueItem]): When nothing matched, the items with
            the closest names.
    """

    model: str
    query: str
    matches: List[CatalogueItem]
    suggestions: List[CatalogueItem]


class LookupRequest(BaseModel):
    """Items to look up.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown fields are rejected.
        items (List[Label]): Ids or names.
    """

    model_config = ConfigDict(extra="forbid")

    items: List[Label] = Field(min_length=1, max_length=1_000)


class UnknownItem(BaseModel):
    """An item a lookup did not find.

    Attributes:
        query (Label): What was asked.
        suggestions (List[str]): The closest names.
    """

    query: Label
    suggestions: List[str]


class LookupResponse(BaseModel):
    """The items a lookup found and the ones it did not.

    Attributes:
        model (str): The endpoint that answered.
        items (List[CatalogueItem]): The items found, in request order.
        unknown (List[UnknownItem]): The items not found.
    """

    model: str
    items: List[CatalogueItem]
    unknown: List[UnknownItem]


class PopularRequest(BaseModel):
    """A request for the most popular items.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown fields are rejected.
        k (Optional[int]): How many items to return.
        filter (Optional[Dict[str, Union[str, List[str]]]]): Item attributes
            the answer must have.
        exclude (List[Label]): Items never to return.
    """

    model_config = ConfigDict(extra="forbid")

    k: Optional[int] = None
    filter: Optional[Dict[str, Union[str, List[str]]]] = None
    exclude: List[Label] = Field(default_factory=list)


class PopularResponse(BaseModel):
    """The most popular items, most interacted first.

    Attributes:
        model (str): The endpoint that answered.
        items (List[ItemScore]): The items.
    """

    model: str
    items: List[ItemScore]


class ContextDescription(BaseModel):
    """The context a context-aware model accepts.

    Attributes:
        model (str): The endpoint that answered.
        fields (Dict[str, Any]): Each field's type, accepted values (most
            frequent first, with counts), range and description.
        example (Dict[str, Any]): A context the model accepts.
    """

    model: str
    fields: Dict[str, Any]
    example: Dict[str, Any]


class ModelInfo(BaseModel):
    """What an endpoint serves.

    Attributes:
        name (str): The endpoint name.
        model (str): The WarpRec model class.
        kind (str): 'general' or 'sequential'.
        n_users (int): Users seen in training.
        n_items (int): Items the model scores.
        needs_user (bool): Whether a history needs a known user alongside it.
        warprec_version (Optional[str]): The WarpRec that saved the model.
        params (Dict[str, Any]): The model's hyperparameters.
        context (Optional[Dict[str, Any]]): For a context-aware model, the
            fields a request must describe and the values each accepts.
        description (Optional[str]): What the endpoint serves, in words.
        item_noun (str): What an item is called.
        how_to_ask (List[str]): What a request must and may contain.
        example_request (Dict[str, Any]): A request the endpoint answers.
        catalogue (Optional[Dict[str, Any]]): What the endpoint knows about its
            items: names, attributes and their most common values.
        training (Optional[Dict[str, Any]]): What the model was trained on and
            how it scored; None for checkpoints saved before it was recorded.
        example_context (Optional[Dict[str, Any]]): For a context-aware model,
            a context it accepts.
    """

    name: str
    model: str
    kind: str
    n_users: int
    n_items: int
    needs_user: bool
    warprec_version: Optional[str] = None
    params: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    item_noun: str = "item"
    how_to_ask: List[str] = Field(default_factory=list)
    example_request: Dict[str, Any] = Field(default_factory=dict)
    catalogue: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    example_context: Optional[Dict[str, Any]] = None
