from itertools import product
from typing import ClassVar

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
    validate_between_zero_and_one,
    validate_layer_list,
    validate_bool_values,
    validate_numeric_values,
    validate_str_list,
)
from warprec.utils.registry import params_registry


@params_registry.register("EGCF")
class EGCF(RecomModel):
    """Definition of the model EGCF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD): List of values for n_layers.
        ssl_lambda (FLOAT_FIELD): List of values for ssl_lambda.
        temperature (FLOAT_FIELD): List of values for temperature.
        mode (STR_FIELD): List of values for mode.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    ssl_lambda: FLOAT_FIELD
    temperature: FLOAT_FIELD
    mode: STR_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("ssl_lambda")
    @classmethod
    def check_ssl_lambda(cls, v: list):
        """Validate ssl_lambda."""
        return validate_greater_equal_than_zero(cls, v, "ssl_lambda")

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: list):
        """Validate temperature."""
        return validate_greater_than_zero(cls, v, "temperature")

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v: list):
        """Validate mode."""
        allowed = ["parallel", "alternating"]
        return validate_str_list(cls, v, allowed, "mode")

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


@params_registry.register("GCMC")
class GCMC(RecomModel):
    """Definition of the model GCMC.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
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


@params_registry.register("LightGCN")
class LightGCN(RecomModel):
    """Definition of the model LightGCN.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD): List of values for n_layers.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
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


@params_registry.register("LightGCNpp")
class LightGCNpp(RecomModel):
    """Definition of the model LightGCNpp.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD): List of values for n_layers.
        alpha (FLOAT_FIELD): List of values for alpha.
        beta (FLOAT_FIELD): List of values for beta.
        gamma (FLOAT_FIELD): List of values for gamma.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    alpha: FLOAT_FIELD
    beta: FLOAT_FIELD
    gamma: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_between_zero_and_one(cls, v, "alpha")

    @field_validator("beta")
    @classmethod
    def check_beta(cls, v: list):
        """Validate beta."""
        return validate_numeric_values(v)

    @field_validator("gamma")
    @classmethod
    def check_gamma(cls, v: list):
        """Validate gamma."""
        return validate_between_zero_and_one(cls, v, "gamma")

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


@params_registry.register("NGCF")
class NGCF(RecomModel):
    """Definition of the model NGCF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        weight_size (LIST_INT_FIELD): List of values for weight sizes.
        node_dropout (FLOAT_FIELD): List of values for node dropout rate.
        message_dropout (FLOAT_FIELD): List of values for message dropout rate.
        reg_weight (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    weight_size: LIST_INT_FIELD
    node_dropout: FLOAT_FIELD
    message_dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

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


@params_registry.register("UltraGCN")
class UltraGCN(RecomModel):
    """Definition of the model UltraGCN.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        w_lambda (FLOAT_FIELD): List of values for w_lambda.
        w_gamma (FLOAT_FIELD): List of values for w_gamma.
        w_neg (FLOAT_FIELD): List of values for w_neg.
        ii_k (INT_FIELD): List of values for ii_k.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    w_lambda: FLOAT_FIELD
    w_gamma: FLOAT_FIELD
    w_neg: FLOAT_FIELD
    ii_k: INT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("w_lambda")
    @classmethod
    def check_w_lambda(cls, v: list):
        return validate_greater_equal_than_zero(cls, v, "w_lambda")

    @field_validator("w_gamma")
    @classmethod
    def check_w_gamma(cls, v: list):
        return validate_greater_equal_than_zero(cls, v, "w_gamma")

    @field_validator("w_neg")
    @classmethod
    def check_w_neg(cls, v: list):
        return validate_greater_than_zero(cls, v, "w_neg")

    @field_validator("ii_k")
    @classmethod
    def check_ii_k(cls, v: list):
        return validate_greater_than_zero(cls, v, "ii_k")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        return validate_greater_than_zero(cls, v, "learning_rate")


@params_registry.register("XSimGCL")
class XSimGCL(RecomModel):
    """Definition of the model XSimGCL.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD): List of values for n_layers.
        lambda_ (FLOAT_FIELD): List of values for lambda (contrastive weight).
        eps (FLOAT_FIELD): List of values for eps (perturbation noise).
        temperature (FLOAT_FIELD): List of values for temperature.
        layer_cl (INT_FIELD): List of values for layer_cl (layer for CL).
        reg_weight (FLOAT_FIELD): List of values for L2 regularization weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        need_single_trial_validation (ClassVar[bool]): Flag to enable single trial validation.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    lambda_: FLOAT_FIELD
    eps: FLOAT_FIELD
    temperature: FLOAT_FIELD
    layer_cl: INT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("lambda_")
    @classmethod
    def check_lambda(cls, v: list):
        """Validate lambda_."""
        return validate_greater_equal_than_zero(cls, v, "lambda_")

    @field_validator("eps")
    @classmethod
    def check_eps(cls, v: list):
        """Validate eps."""
        return validate_greater_equal_than_zero(cls, v, "eps")

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: list):
        """Validate temperature."""
        return validate_greater_than_zero(cls, v, "temperature")

    @field_validator("layer_cl")
    @classmethod
    def check_layer_cl(cls, v: list):
        """Validate layer_cl."""
        return validate_greater_than_zero(cls, v, "layer_cl")

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

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.

        Ensures that there is at least one combination where layer_cl <= n_layers.
        """
        n_layers_list = self._clean_param_list(self.n_layers)
        layer_cl_list = self._clean_param_list(self.layer_cl)

        has_valid_combination = any(
            cl_layer <= n_layer
            for n_layer, cl_layer in product(n_layers_list, layer_cl_list)
        )

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for XSimGCL. "
                "Ensure there's at least one combination where 'layer_cl' "
                "is less than or equal to 'n_layers'."
            )

    def validate_single_trial_params(self):
        """Validates the coherence of n_layers and layer_cl for a single trial."""
        n_layers_clean = (
            self.n_layers[1]
            if self.n_layers and isinstance(self.n_layers[0], str)
            else self.n_layers[0]
        )
        layer_cl_clean = (
            self.layer_cl[1]
            if self.layer_cl and isinstance(self.layer_cl[0], str)
            else self.layer_cl[0]
        )

        if layer_cl_clean > n_layers_clean:
            raise ValueError(
                f"Inconsistent configuration for XSimGCL: "
                f"layer_cl ({layer_cl_clean}) cannot be greater than "
                f"n_layers ({n_layers_clean})."
            )
