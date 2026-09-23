from typing import ClassVar, Optional

from pydantic import field_validator

from warprec.utils.config.common import (
    validate_between_zero_and_one,
    validate_greater_equal_than_zero,
    validate_greater_than_zero,
    validate_value_list,
)
from warprec.utils.config.model_configuration import (
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FLOAT_FIELD,
    LIST_STR_FIELD,
    RecomModel,
)
from warprec.utils.registry import params_registry


@params_registry.register("VBPR")
class VBPR(RecomModel):
    """Definition of the model VBPR.

    Attributes:
        need_multimodal (ClassVar[bool]): The model scores from item features.
        embedding_size (INT_FIELD): List of values for embedding_size.
        modalities (Optional[LIST_STR_FIELD]): The modalities to read. Defaults
            to every configured one.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_multimodal: ClassVar[bool] = True

    embedding_size: INT_FIELD
    modalities: Optional[LIST_STR_FIELD] = None
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("modalities")
    @classmethod
    def check_modalities(cls, v: Optional[list]):
        """Validate modalities."""
        return v if v is None else validate_value_list(cls, v, "modalities")

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


@params_registry.register("FREEDOM")
class FREEDOM(RecomModel):
    """Definition of the model FREEDOM.

    Attributes:
        need_multimodal (ClassVar[bool]): The model scores from item features.
        embedding_size (INT_FIELD): List of values for embedding_size.
        feature_size (INT_FIELD): List of values for feature_size.
        knn_k (INT_FIELD): List of values for the size of a neighbourhood.
        n_layers (INT_FIELD): List of values for the item-item hops.
        n_ui_layers (INT_FIELD): List of values for the user-item hops.
        dropout (FLOAT_FIELD): List of values for the share of dropped edges.
        modalities (Optional[LIST_STR_FIELD]): The modalities to read. Defaults
            to every configured one.
        modality_weights (Optional[LIST_FLOAT_FIELD]): How much each modality's
            item-item graph counts. Defaults to equal weight.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_multimodal: ClassVar[bool] = True

    embedding_size: INT_FIELD
    feature_size: INT_FIELD
    knn_k: INT_FIELD
    n_layers: INT_FIELD
    n_ui_layers: INT_FIELD
    dropout: FLOAT_FIELD
    modalities: Optional[LIST_STR_FIELD] = None
    modality_weights: Optional[LIST_FLOAT_FIELD] = None
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("feature_size")
    @classmethod
    def check_feature_size(cls, v: list):
        """Validate feature_size."""
        return validate_greater_than_zero(cls, v, "feature_size")

    @field_validator("knn_k")
    @classmethod
    def check_knn_k(cls, v: list):
        """Validate knn_k."""
        return validate_greater_than_zero(cls, v, "knn_k")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_equal_than_zero(cls, v, "n_layers")

    @field_validator("n_ui_layers")
    @classmethod
    def check_n_ui_layers(cls, v: list):
        """Validate n_ui_layers."""
        return validate_greater_equal_than_zero(cls, v, "n_ui_layers")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_between_zero_and_one(cls, v, "dropout")

    @field_validator("modalities")
    @classmethod
    def check_modalities(cls, v: Optional[list]):
        """Validate modalities."""
        return v if v is None else validate_value_list(cls, v, "modalities")

    @field_validator("modality_weights")
    @classmethod
    def check_modality_weights(cls, v: Optional[list]):
        """Validate modality_weights."""
        return v if v is None else validate_value_list(cls, v, "modality_weights")

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
