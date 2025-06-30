from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
    LIST_INT_FIELD,
    BOOL_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_between_zero_and_one,
    validate_layer_list,
    validate_bool_values,
)
from warprec.utils.registry import params_registry


@params_registry.register("LightGCN")
class LightGCN(RecomModel):
    """Definition of the model LightGCN.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD): List of values for n_layers.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    reg_weight: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_k(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("reg_weight")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

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


@params_registry.register("NGCF")
class NGCF(RecomModel):
    """Definition of the model NGCF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        weight_size (LIST_INT_FIELD): List of values for weight sizes.
        node_dropout (FLOAT_FIELD): List of values for node dropout rate.
        message_dropout (FLOAT_FIELD): List of values for message dropout rate.
    """

    embedding_size: INT_FIELD
    reg_weight: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    weight_size: LIST_INT_FIELD
    node_dropout: FLOAT_FIELD
    message_dropout: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

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

    @field_validator("weight_size")
    @classmethod
    def check_weight_size(cls, v: list):
        """Validate weight_size."""
        return validate_layer_list(cls, v, "weight_size")

    @field_validator("node_dropout")
    @classmethod
    def check_node_dropout(cls, v: list):
        """Validate node_dropout."""
        return validate_between_zero_and_one(cls, v, "node_dropout")

    @field_validator("message_dropout")
    @classmethod
    def check_message_dropout(cls, v: list):
        """Validate message_dropout."""
        return validate_between_zero_and_one(cls, v, "message_dropout")


@params_registry.register("RP3Beta")
class RP3Beta(RecomModel):
    """Definition of the model RP3Beta.

    Attributes:
        k (INT_FIELD): List of values for k.
        alpha (FLOAT_FIELD): List of values for alpha.
        beta (FLOAT_FIELD): List of values for beta.
        normalize (BOOL_FIELD): List of values for normalize.
    """

    k: INT_FIELD
    alpha: FLOAT_FIELD
    beta: FLOAT_FIELD
    normalize: BOOL_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        return validate_greater_than_zero(cls, v, "k")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")

    @field_validator("beta")
    @classmethod
    def check_beta(cls, v: list):
        """Validate beta."""
        return validate_greater_equal_than_zero(cls, v, "beta")

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        return validate_bool_values(v)
