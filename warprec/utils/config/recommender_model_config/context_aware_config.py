from pydantic import field_validator

from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
    LIST_INT_FIELD,
    BOOL_FIELD,
    STR_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_layer_list,
    validate_bool_values,
    validate_str_list,
)
from warprec.utils.registry import params_registry


@params_registry.register("AFM")
class AFM(RecomModel):
    """Definition of the model AFM.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        attention_size (INT_FIELD): List of attention_size values.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    attention_size: INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("attention_size")
    @classmethod
    def check_attention_size(cls, v: list):
        """Validate attention_size."""
        return validate_greater_than_zero(cls, v, "attention_size")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("DCN")
class DCN(RecomModel):
    """Definition of the model DCN.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        cross_layer_num (INT_FIELD): List of cross_layer_num values.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    cross_layer_num: INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("cross_layer_num")
    @classmethod
    def check_cross_layer_num(cls, v: list):
        """Validate cross_layer_num."""
        return validate_greater_than_zero(cls, v, "cross_layer_num")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("DCNv2")
class DCNv2(RecomModel):
    """Definition of the model DCNv2.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        cross_layer_num (INT_FIELD): List of cross_layer_num values.
        dropout (FLOAT_FIELD): List of values for dropout.
        model_structure (STR_FIELD): List of model structures.
        use_mixed (BOOL_FIELD): List of boolean values for Mixed Cross Network usage.
        expert_num (INT_FIELD): List of number of experts.
        low_rank (INT_FIELD): List of low rank dimensions.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    cross_layer_num: INT_FIELD
    dropout: FLOAT_FIELD
    model_structure: STR_FIELD
    use_mixed: BOOL_FIELD
    expert_num: INT_FIELD
    low_rank: INT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("cross_layer_num")
    @classmethod
    def check_cross_layer_num(cls, v: list):
        """Validate cross_layer_num."""
        return validate_greater_than_zero(cls, v, "cross_layer_num")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("model_structure")
    @classmethod
    def check_model_structure(cls, v: list):
        """Validate model_structure."""
        allowed = ["parallel", "stacked"]
        return validate_str_list(cls, v, allowed, "model_structure")

    @field_validator("use_mixed")
    @classmethod
    def check_use_mixed(cls, v: list):
        """Validate use_mixed."""
        return validate_bool_values(v)

    @field_validator("expert_num")
    @classmethod
    def check_expert_num(cls, v: list):
        """Validate expert_num."""
        return validate_greater_than_zero(cls, v, "expert_num")

    @field_validator("low_rank")
    @classmethod
    def check_low_rank(cls, v: list):
        """Validate low_rank."""
        return validate_greater_than_zero(cls, v, "low_rank")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("DeepFM")
class DeepFM(RecomModel):
    """Definition of the model DeepFM.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("FM")
class FM(RecomModel):
    """Definition of the model FM.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("NFM")
class NFM(RecomModel):
    """Definition of the model NFM.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("WideAndDeep")
class WideAndDeep(RecomModel):
    """Definition of the model WideAndDeep.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")


@params_registry.register("xDeepFM")
class xDeepFM(RecomModel):
    """Definition of the model xDeepFM.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        cin_layer_size (LIST_INT_FIELD): List of cin_layer_size values.
        dropout (FLOAT_FIELD): List of values for dropout.
        direct (BOOL_FIELD): List of values for direct.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
    """

    embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    cin_layer_size: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    direct: BOOL_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("cin_layer_size")
    @classmethod
    def check_cin_layer_size(cls, v: list):
        """Validate cin_layer_size."""
        return validate_layer_list(cls, v, "cin_layer_size")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("direct")
    @classmethod
    def check_direct(cls, v: list):
        """Validate direct."""
        return validate_bool_values(v)

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")
