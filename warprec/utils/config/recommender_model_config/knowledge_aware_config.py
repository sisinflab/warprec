from typing import ClassVar

from pydantic import field_validator

from warprec.utils.config.common import (
    validate_layer_list,
    validate_greater_equal_than_zero,
    validate_greater_than_zero,
)
from warprec.utils.config.model_configuration import (
    FLOAT_FIELD,
    INT_FIELD,
    LIST_INT_FIELD,
    RecomModel,
)
from warprec.utils.registry import params_registry


@params_registry.register("CKE")
class CKE(RecomModel):
    """Definition of the model CKE.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        kg_embedding_size (INT_FIELD): List of values for kg_embedding_size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        kg_reg_weight (FLOAT_FIELD): List of values for kg_reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    kg_embedding_size: INT_FIELD
    reg_weight: FLOAT_FIELD
    kg_reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("kg_embedding_size")
    @classmethod
    def check_kg_embedding_size(cls, v: list):
        """Validate kg_embedding_size."""
        return validate_greater_than_zero(cls, v, "kg_embedding_size")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("kg_reg_weight")
    @classmethod
    def check_kg_reg_weight(cls, v: list):
        """Validate kg_reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "kg_reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

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


@params_registry.register("KGAT")
class KGAT(RecomModel):
    """Definition of the model KGAT.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        kg_embedding_size (INT_FIELD): List of values for kg_embedding_size.
        layers (LIST_INT_FIELD): List of propagation layer widths.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    kg_embedding_size: INT_FIELD
    layers: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("kg_embedding_size")
    @classmethod
    def check_kg_embedding_size(cls, v: list):
        """Validate kg_embedding_size."""
        return validate_greater_than_zero(cls, v, "kg_embedding_size")

    @field_validator("layers")
    @classmethod
    def check_layers(cls, v: list):
        """Validate layers."""
        return validate_layer_list(cls, v, "layers")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

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
