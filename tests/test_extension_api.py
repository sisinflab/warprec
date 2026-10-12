"""The extension points a user writes against: models, parameters, metrics, stash.

Each test pins down a trap a user fell into: something that silently did the
wrong thing, or failed far from its cause, now either works or says what to do.
"""

import uuid

from warprec.recommenders.base_recommender import Recommender
from warprec.recommenders.trainer.trainer import Trainer
from warprec.utils.config.model_configuration import INT_FIELD, RecomModel
from warprec.utils.enums import SearchSpace
from warprec.utils.registry import model_registry, params_registry


def unique_name(prefix: str) -> str:
    """A registry name no other test uses.

    Args:
        prefix (str): The readable start of the name.

    Returns:
        str: The prefix followed by a random suffix.
    """
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def test_a_parameter_class_may_be_named_unlike_its_model():
    """The class is found by the name it is registered under, not its own."""
    name = unique_name("MyBPR")

    @model_registry.register(name)
    class MyModel(Recommender):  # pylint: disable=unused-variable
        """A model the parameter class below describes."""

        embedding_size: int

    @params_registry.register(name)
    class MyModelParams(RecomModel):
        """Parameters named after nothing in particular."""

        embedding_size: INT_FIELD

    params = MyModelParams(embedding_size=[8, 16])

    assert params.embedding_size == [SearchSpace.GRID, 8, 16]
    space = Trainer._parse_params(None, params)  # type: ignore[arg-type]
    assert list(space["embedding_size"]["grid_search"]) == [8, 16]


def test_a_parameter_class_named_like_its_model_still_validates():
    """The documented naming keeps working."""
    name = unique_name("Named")

    @model_registry.register(name)
    class Model(Recommender):  # pylint: disable=unused-variable
        """A model the parameter class below describes."""

        embedding_size: int

    params_class = type(
        name, (RecomModel,), {"__annotations__": {"embedding_size": INT_FIELD}}
    )
    params_registry.register(name)(params_class)

    params = params_class(embedding_size=[8, 16])

    assert params.embedding_size == [SearchSpace.GRID, 8, 16]
