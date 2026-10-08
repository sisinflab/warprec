from pydantic import field_validator

from warprec.utils.config.common import validate_greater_than_zero
from warprec.utils.config.model_configuration import (
    FLOAT_FIELD,
    INT_FIELD,
    RecomModel,
)
from warprec.utils.registry import params_registry


# WarpRec looks a parameter class up by its own class name, so it must be
# named exactly like the model it validates. Hence a module of its own.
@params_registry.register("MyBPR")
class MyBPR(RecomModel):
    """The hyperparameters of the MyBPR model.

    Each field accepts a single value or a search space (guide 10).

    Attributes:
        embedding_size (INT_FIELD): The size of the user and item factors.
        batch_size (INT_FIELD): The number of training triples per batch.
        epochs (INT_FIELD): The number of passes over the training data.
        learning_rate (FLOAT_FIELD): The optimiser's learning rate.
    """

    embedding_size: INT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size", "batch_size", "epochs", "learning_rate")
    @classmethod
    def check_positive(cls, v, info):
        """Every hyperparameter of MyBPR must be greater than zero."""
        return validate_greater_than_zero(cls, v, info.field_name)
