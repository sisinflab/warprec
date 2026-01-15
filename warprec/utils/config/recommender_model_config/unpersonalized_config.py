# pylint: disable=duplicate-code
from warprec.utils.config.model_configuration import RecomModel
from warprec.utils.registry import params_registry


@params_registry.register("Pop")
class Pop(RecomModel):
    """Empty definition of the model Pop."""


@params_registry.register("Random")
class Random(RecomModel):
    """Empty definition of the model Random."""
