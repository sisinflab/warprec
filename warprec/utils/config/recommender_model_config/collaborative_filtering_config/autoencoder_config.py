from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    BOOL_FIELD,
    LIST_INT_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_str_list,
    validate_between_zero_and_one,
    validate_bool_values,
    validate_layer_list,
)
from warprec.utils.registry import params_registry


@params_registry.register("CDAE")
class CDAE(RecomModel):
    """Definition of the model CDAE.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        corruption (FLOAT_FIELD): List of values for corruption.
        hid_activation (STR_FIELD): List of activation functions for hidden layer.
        out_activation (STR_FIELD): List of activation functions for output layer.
        loss_type (STR_FIELD): List of loss types to use.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    corruption: FLOAT_FIELD
    hid_activation: STR_FIELD
    out_activation: STR_FIELD
    loss_type: STR_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("corruption")
    @classmethod
    def check_corruption(cls, v: list):
        """Validate corruption."""
        return validate_greater_equal_than_zero(cls, v, "corruption")

    @field_validator("hid_activation")
    @classmethod
    def check_hid_activation(cls, v: list):
        """Validate hid_activation."""
        allowed = ["relu", "tanh", "sigmoid"]
        return validate_str_list(cls, v, allowed, "hid_activation")

    @field_validator("out_activation")
    @classmethod
    def check_out_activation(cls, v: list):
        """Validate out_activation."""
        allowed = ["relu", "sigmoid"]
        return validate_str_list(cls, v, allowed, "out_activation")

    @field_validator("loss_type")
    @classmethod
    def check_loss_type(cls, v: list):
        """Validate loss_type."""
        allowed = ["MSE", "BCE"]
        return validate_str_list(cls, v, allowed, "loss_type")

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


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (FLOAT_FIELD): List of values that l2 regularization can take.
    """

    l2: FLOAT_FIELD

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v: list):
        """Validate l2."""
        return validate_greater_than_zero(cls, v, "l2")


@params_registry.register("MacridVAE")
class MacridVAE(RecomModel):
    """Definition of the model MacridVAE.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        encoder_hidden_dims (LIST_INT_FIELD): List of values for encoder_hidden_dims.
        k_fac (INT_FIELD): List of values for k_fac (number of concepts).
        tau (FLOAT_FIELD): List of values for tau (temperature).
        corruption (FLOAT_FIELD): List of values for corruption (dropout).
        nogb (BOOL_FIELD): List of values for nogb (No Gumbel-Softmax).
        std (FLOAT_FIELD): List of values for std (prior std deviation).
        anneal_cap (FLOAT_FIELD): List of values for anneal_cap.
        total_anneal_steps (INT_FIELD): List of values for total_anneal_steps.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    encoder_hidden_dims: LIST_INT_FIELD
    k_fac: INT_FIELD
    tau: FLOAT_FIELD
    corruption: FLOAT_FIELD
    nogb: BOOL_FIELD
    std: FLOAT_FIELD
    anneal_cap: FLOAT_FIELD
    total_anneal_steps: INT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("encoder_hidden_dims")
    @classmethod
    def check_encoder_hidden_dims(cls, v: list):
        """Validate encoder_hidden_dims."""
        return validate_layer_list(cls, v, "encoder_hidden_dims")

    @field_validator("k_fac")
    @classmethod
    def check_k_fac(cls, v: list):
        """Validate k_fac."""
        return validate_greater_than_zero(cls, v, "k_fac")

    @field_validator("tau")
    @classmethod
    def check_tau(cls, v: list):
        """Validate tau."""
        return validate_greater_than_zero(cls, v, "tau")

    @field_validator("corruption")
    @classmethod
    def check_corruption(cls, v: list):
        """Validate corruption."""
        return validate_between_zero_and_one(cls, v, "corruption")

    @field_validator("nogb")
    @classmethod
    def check_nogb(cls, v: list):
        """Validate nogb."""
        return validate_bool_values(v)

    @field_validator("std")
    @classmethod
    def check_std(cls, v: list):
        """Validate std."""
        return validate_greater_than_zero(cls, v, "std")

    @field_validator("anneal_cap")
    @classmethod
    def check_anneal_cap(cls, v: list):
        """Validate anneal_cap."""
        return validate_between_zero_and_one(cls, v, "anneal_cap")

    @field_validator("total_anneal_steps")
    @classmethod
    def check_total_anneal_steps(cls, v: list):
        """Validate total_anneal_steps."""
        return validate_greater_than_zero(cls, v, "total_anneal_steps")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
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


@params_registry.register("MultiDAE")
class MultiDAE(RecomModel):
    """Definition of the model MultiDAE.

    Attributes:
        intermediate_dim (INT_FIELD): List of intermediate_dim values.
        latent_dim (INT_FIELD): List of values for latent_dim values.
        corruption (FLOAT_FIELD): List of values for corruption.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    intermediate_dim: INT_FIELD
    latent_dim: INT_FIELD
    corruption: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("intermediate_dim")
    @classmethod
    def check_intermediate_dim(cls, v: list):
        """Validate intermediate_dim."""
        return validate_greater_than_zero(cls, v, "intermediate_dim")

    @field_validator("latent_dim")
    @classmethod
    def check_latent_dim(cls, v: list):
        """Validate latent_dim."""
        return validate_greater_than_zero(cls, v, "latent_dim")

    @field_validator("corruption")
    @classmethod
    def check_corruption(cls, v: list):
        """Validate corruption."""
        return validate_greater_equal_than_zero(cls, v, "corruption")

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


@params_registry.register("MultiVAE")
class MultiVAE(RecomModel):
    """Definition of the model MultiVAE.

    Attributes:
        intermediate_dim (INT_FIELD): List of intermediate_dim values.
        latent_dim (INT_FIELD): List of values for latent_dim values.
        corruption (FLOAT_FIELD): List of values for corruption.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        anneal_cap (FLOAT_FIELD): List of values for anneal_cap.
        anneal_step (INT_FIELD): List of values for anneal_step.
    """

    intermediate_dim: INT_FIELD
    latent_dim: INT_FIELD
    corruption: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    anneal_cap: FLOAT_FIELD
    anneal_step: INT_FIELD

    @field_validator("intermediate_dim")
    @classmethod
    def check_intermediate_dim(cls, v: list):
        """Validate intermediate_dim."""
        return validate_greater_than_zero(cls, v, "intermediate_dim")

    @field_validator("latent_dim")
    @classmethod
    def check_latent_dim(cls, v: list):
        """Validate latent_dim."""
        return validate_greater_than_zero(cls, v, "latent_dim")

    @field_validator("corruption")
    @classmethod
    def check_corruption(cls, v: list):
        """Validate corruption."""
        return validate_greater_equal_than_zero(cls, v, "corruption")

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

    @field_validator("anneal_cap")
    @classmethod
    def check_anneal_cap(cls, v: list):
        """Validate anneal_cap."""
        return validate_greater_equal_than_zero(cls, v, "anneal_cap")

    @field_validator("anneal_step")
    @classmethod
    def check_anneal_step(cls, v: list):
        """Validate anneal_step."""
        return validate_greater_equal_than_zero(cls, v, "anneal_step")
