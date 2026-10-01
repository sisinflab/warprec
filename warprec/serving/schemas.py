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
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[Label] = None
    history: Optional[List[Label]] = None
    k: Optional[int] = None
    exclude: List[Label] = Field(default_factory=list)
    context: Optional[Union[Dict[str, Any], List[Any]]] = None


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
        score (float): The model's score.
        name (Optional[str]): The item's name, when a catalogue is configured.
    """

    item_id: Label
    score: float
    name: Optional[str] = None


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
