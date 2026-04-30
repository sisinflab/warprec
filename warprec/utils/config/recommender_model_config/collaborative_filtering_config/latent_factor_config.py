# pylint: disable=duplicate-code
from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    STR_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_bool_values,
    validate_between_zero_and_one,
)
from warprec.utils.registry import params_registry


@params_registry.register("ADMMSlim")
class ADMMSlim(RecomModel):
    """Definition of the model ADMMSlim.

    Attributes:
        lambda_1 (FLOAT_FIELD): List of values for lambda_1.
        lambda_2 (FLOAT_FIELD): List of values for lambda_2.
        alpha (FLOAT_FIELD): List of values for alpha.
        rho (FLOAT_FIELD): List of values for rho.
        it (INT_FIELD): List of values for it.
        positive_only (BOOL_FIELD): List of values for positive_only.
        center_columns (BOOL_FIELD): List of values for center_columns.
    """

    lambda_1: FLOAT_FIELD
    lambda_2: FLOAT_FIELD
    alpha: FLOAT_FIELD
    rho: FLOAT_FIELD
    it: INT_FIELD
    positive_only: BOOL_FIELD
    center_columns: BOOL_FIELD

    @field_validator("lambda_1")
    @classmethod
    def check_lambda_1(cls, v: list):
        """Validate lambda_1."""
        return validate_greater_equal_than_zero(cls, v, "lambda_1")

    @field_validator("lambda_2")
    @classmethod
    def check_lambda_2(cls, v: list):
        """Validate lambda_2."""
        return validate_greater_equal_than_zero(cls, v, "lambda_2")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")

    @field_validator("rho")
    @classmethod
    def check_rho(cls, v: list):
        """Validate rho."""
        return validate_greater_equal_than_zero(cls, v, "rho")

    @field_validator("it")
    @classmethod
    def check_it(cls, v: list):
        """Validate it."""
        return validate_greater_equal_than_zero(cls, v, "it")

    @field_validator("positive_only")
    @classmethod
    def check_positive_only(cls, v: list):
        """Validate positive_only."""
        return validate_bool_values(v)

    @field_validator("center_columns")
    @classmethod
    def check_center_columns(cls, v: list):
        """Validate center_columns."""
        return validate_bool_values(v)


@params_registry.register("BPR")
class BPR(RecomModel):
    """Definition of the model BPR.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    reg_weight: FLOAT_FIELD
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


@params_registry.register("FISM")
class FISM(RecomModel):
    """Definition of the model FISM.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        alpha (FLOAT_FIELD): List of values for alpha.
        split_to (INT_FIELD): List of values for split_to.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    alpha: FLOAT_FIELD
    split_to: INT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")

    @field_validator("split_to")
    @classmethod
    def check_split_to(cls, v: list):
        """Validate split_to."""
        return validate_greater_than_zero(cls, v, "split_to")

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


@params_registry.register("iALS")
class iALS(RecomModel):
    """Definition of the model iALS.

    Attributes:
         factors (INT_FIELD): List of values for factors.
         alpha0 (FLOAT_FIELD): List of values for alpha0.
         reg (FLOAT_FIELD): List of values for reg.
         n_iterations (INT_FIELD): List of values for n_iterations.
         nu (FLOAT_FIELD): List of values for nu.
    """

    factors: INT_FIELD
    alpha0: FLOAT_FIELD
    reg: FLOAT_FIELD
    n_iterations: INT_FIELD
    nu: FLOAT_FIELD

    @field_validator("factors")
    @classmethod
    def check_factors(cls, v: list):
        """Validate factors."""
        return validate_greater_than_zero(cls, v, "factors")

    @field_validator("alpha0")
    @classmethod
    def check_alpha0(cls, v: list):
        """Validate alpha0."""
        return validate_greater_equal_than_zero(cls, v, "alpha0")

    @field_validator("reg")
    @classmethod
    def check_reg(cls, v: list):
        """Validate reg."""
        return validate_greater_equal_than_zero(cls, v, "reg")

    @field_validator("n_iterations")
    @classmethod
    def check_n_iterations(cls, v: list):
        """Validate n_iterations."""
        return validate_greater_than_zero(cls, v, "n_iterations")

    @field_validator("nu")
    @classmethod
    def check_nu(cls, v: list):
        """Validate nu."""
        return validate_greater_equal_than_zero(cls, v, "nu")


@params_registry.register("iALS2008")
class iALS2008(RecomModel):
    """Definition of the model iALS2008.

    Attributes:
        factors (INT_FIELD): List of values for factors.
        alpha (FLOAT_FIELD): List of values for alpha.
        reg (FLOAT_FIELD): List of values for reg.
        n_iterations (INT_FIELD): List of values for n_iterations.
        confidence_type (STR_FIELD): List of values for confidence_type.
        epsilon (FLOAT_FIELD): List of values for epsilon.
    """

    factors: INT_FIELD
    alpha: FLOAT_FIELD
    reg: FLOAT_FIELD
    n_iterations: INT_FIELD
    confidence_type: STR_FIELD
    epsilon: FLOAT_FIELD

    @field_validator("factors")
    @classmethod
    def check_factors(cls, v: list):
        """Validate factors."""
        return validate_greater_than_zero(cls, v, "factors")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")

    @field_validator("reg")
    @classmethod
    def check_reg(cls, v: list):
        """Validate reg."""
        return validate_greater_equal_than_zero(cls, v, "reg")

    @field_validator("n_iterations")
    @classmethod
    def check_n_iterations(cls, v: list):
        """Validate n_iterations."""
        return validate_greater_than_zero(cls, v, "n_iterations")

    @field_validator("confidence_type")
    @classmethod
    def check_confidence_type(cls, v: list):
        """Validate confidence_type."""
        valid_values = {"linear", "log"}
        for val in v:
            if val not in valid_values:
                raise ValueError(
                    f"Invalid confidence_type: {val}. Must be one of {valid_values}."
                )
        return v

    @field_validator("epsilon")
    @classmethod
    def check_epsilon(cls, v: list):
        """Validate epsilon."""
        return validate_greater_than_zero(cls, v, "epsilon")


@params_registry.register("MACRMF")
class MACRMF(RecomModel):
    """Definition of the model MACRMF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        alpha (FLOAT_FIELD): List of values for alpha.
        beta (FLOAT_FIELD): List of values for beta.
        c (FLOAT_FIELD): List of values for c.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        neg_samples (INT_FIELD): List of values for neg_samples.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    alpha: FLOAT_FIELD
    beta: FLOAT_FIELD
    c: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    neg_samples: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

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

    @field_validator("c")
    @classmethod
    def check_c(cls, v: list):
        """Validate c."""
        return validate_greater_equal_than_zero(cls, v, "c")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_than_zero(cls, v, "neg_samples")

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


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (FLOAT_FIELD): List of values that l1 regularization can take.
        alpha (FLOAT_FIELD): List of values that alpha can take.
    """

    l1: FLOAT_FIELD
    alpha: FLOAT_FIELD

    @field_validator("l1")
    @classmethod
    def check_l1(cls, v: list):
        """Validate l1."""
        return validate_between_zero_and_one(cls, v, "l1")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")
