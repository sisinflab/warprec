from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
)
from warprec.utils.registry import params_registry


@params_registry.register("GRU4Rec")
class GRU4Rec(RecomModel):
    """Definition of the model GRU4Rec.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        hidden_size (INT_FIELD):  List of values for hidden_size.
        num_layers (INT_FIELD): List of values for num_layers.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
    """

    embedding_size: INT_FIELD
    hidden_size: INT_FIELD
    num_layers: INT_FIELD
    dropout_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD
    need_timestamp: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("hidden_size")
    @classmethod
    def check_hidden_size(cls, v: list):
        """Validate hidden_size."""
        return validate_greater_than_zero(cls, v, "hidden_size")

    @field_validator("num_layers")
    @classmethod
    def check_num_layers(cls, v: list):
        """Validate num_layers."""
        return validate_greater_than_zero(cls, v, "num_layers")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_greater_equal_than_zero(cls, v, "dropout_prob")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")
