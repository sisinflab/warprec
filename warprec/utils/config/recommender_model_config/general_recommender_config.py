from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    LIST_INT_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    BOOL_FIELD,
)
from warprec.utils.config.common import (
    convert_to_list,
    check_less_equal_zero,
    check_zero_to_one,
    check_less_than_zero,
    check_similarity,
    check_user_profile,
)
from warprec.utils.registry import params_registry, similarities_registry


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
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    f"Values of l2 for EASE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v


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
        v = convert_to_list(v)
        for value in v:
            if not check_zero_to_one(value):
                raise ValueError(
                    "Values of l1 for Slim model must be in [0, 1] range. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of alpha for Slim model must be >= 0. "
                    "In case of alpha=0, ordinary least square will be solved. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("ItemKNN")
class ItemKNN(RecomModel):
    """Definition of the model ItemKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        normalize (BOOL_FIELD): List of values for normalization flag.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    normalize: BOOL_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of k for ItemKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        v = convert_to_list(v)
        for value in v:
            if not check_similarity(value):
                raise ValueError(
                    "Values of similarity for ItemKNN model must be supported similarities. "
                    f"Values received as input: {v}. "
                    f"Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        v = convert_to_list(v)
        return v


@params_registry.register("UserKNN")
class UserKNN(RecomModel):
    """Definition of the model UserKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        normalize (BOOL_FIELD): List of values for normalization flag.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    normalize: BOOL_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of k for UserKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        v = convert_to_list(v)
        for value in v:
            if not check_similarity(value):
                raise ValueError(
                    "Values of similarity for UserKNN model must be supported similarities. "
                    f"Values received as input: {v}. "
                    f"Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        v = convert_to_list(v)
        return v


@params_registry.register("NeuMF")
class NeuMF(RecomModel):
    """Definition of the model NeuMF.

    Attributes:
        mf_embedding_size (INT_FIELD): List of mf embedding size.
        mlp_embedding_size (INT_FIELD): List of mlp embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        mf_train (BOOL_FIELD): List of values for mf_train flag.
        mlp_train (BOOL_FIELD): List of values for mlp_train flag.
        dropout (FLOAT_FIELD): List of values for dropout.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for negative sampling.
    """

    mf_embedding_size: INT_FIELD
    mlp_embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    mf_train: BOOL_FIELD
    mlp_train: BOOL_FIELD
    dropout: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("mf_embedding_size")
    @classmethod
    def check_mf_embedding_size(cls, v: list):
        """Validate mf_embedding_size."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of mf_embedding_size for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("mlp_embedding_size")
    @classmethod
    def check_mlp_embedding_size(cls, v: list):
        """Validate mlp_embedding_size."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of mlp_embedding_size for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        strat = None

        # This should be a list of lists
        # for a more precise validation we do not
        # use the common methods
        if not isinstance(v, list):
            v = [v]
        if not isinstance(v[-1], list):
            v = [v]
        if isinstance(v[0], str):
            strat = v.pop(0)
        for hidden_size in v:
            for value in hidden_size:
                if value <= 0:
                    raise ValueError(
                        "Hidden size for MLP must be > 0. "
                        f"Values received as input: {v}"
                    )
        if strat:
            v.insert(0, strat)
        return v

    @field_validator("mf_train")
    @classmethod
    def check_mf_train(cls, v: list):
        """Validate mf_train."""
        v = convert_to_list(v)
        return v

    @field_validator("mlp_train")
    @classmethod
    def check_mlp_train(cls, v: list):
        """Validate mlp_train."""
        v = convert_to_list(v)
        return v

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of dropout for NeuMF model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of epochs for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of learning_rate for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of neg_samples for NeuMF model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v


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
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of k for RP3Beta model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of alpha for RP3Beta model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("beta")
    @classmethod
    def check_beta(cls, v: list):
        """Validate beta."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of beta for RP3Beta model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        v = convert_to_list(v)
        return v


@params_registry.register("BPR")
class BPR(RecomModel):
    """Definition of the model BPR.

    Attributes:
        embedding_size (INT_FIELD): List of embedding size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of embedding_size for BPR model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of epochs for BPR model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of learning_rate for BPR model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("MultiDAE")
class MultiDAE(RecomModel):
    """Definition of the model MultiDAE.

    Attributes:
        intermediate_dim (INT_FIELD): List of intermediate_dim values.
        latent_dim (INT_FIELD): List of values for latent_dim values.
        dropout (FLOAT_FIELD): List of values for dropout.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        l2_lambda (FLOAT_FIELD): List of values for l2_lambda.
    """

    intermediate_dim: INT_FIELD
    latent_dim: INT_FIELD
    dropout: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    l2_lambda: FLOAT_FIELD

    @field_validator("intermediate_dim")
    @classmethod
    def check_intermediate_dim(cls, v: list):
        """Validate intermediate_dim."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of intermediate_dim for MultiDAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("latent_dim")
    @classmethod
    def check_latent_dim(cls, v: list):
        """Validate latent_dim."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of latent_dim for MultiDAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of dropout for MultiDAE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of epochs for MultiDAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of learning_rate for MultiDAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("l2_lambda")
    @classmethod
    def check_l2_lambda(cls, v: list):
        """Validate l2_lambda."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of l2_lambda for MultiDAE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("MultiVAE")
class MultiVAE(RecomModel):
    """Definition of the model MultiVAE.

    Attributes:
        intermediate_dim (INT_FIELD): List of intermediate_dim values.
        latent_dim (INT_FIELD): List of values for latent_dim values.
        dropout (FLOAT_FIELD): List of values for dropout.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        l2_lambda (FLOAT_FIELD): List of values for l2_lambda.
        anneal_cap (FLOAT_FIELD): List of values for anneal_cap.
        anneal_step (INT_FIELD): List of values for anneal_step.
    """

    intermediate_dim: INT_FIELD
    latent_dim: INT_FIELD
    dropout: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    l2_lambda: FLOAT_FIELD
    anneal_cap: FLOAT_FIELD
    anneal_step: INT_FIELD

    @field_validator("intermediate_dim")
    @classmethod
    def check_intermediate_dim(cls, v: list):
        """Validate intermediate_dim."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of intermediate_dim for MultiVAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("latent_dim")
    @classmethod
    def check_latent_dim(cls, v: list):
        """Validate latent_dim."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of latent_dim for MultiVAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of dropout for MultiVAE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of epochs for MultiVAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of learning_rate for MultiVAE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("l2_lambda")
    @classmethod
    def check_l2_lambda(cls, v: list):
        """Validate l2_lambda."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of l2_lambda for MultiVAE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("anneal_cap")
    @classmethod
    def check_anneal_cap(cls, v: list):
        """Validate anneal_cap."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of anneal_cap for MultiVAE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("anneal_step")
    @classmethod
    def check_anneal_step(cls, v: list):
        """Validate anneal_step."""
        v = convert_to_list(v)
        return v


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
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of lambda_1 for ADMMSlim model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("lambda_2")
    @classmethod
    def check_lambda_2(cls, v: list):
        """Validate lambda_2."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of lambda_2 for ADMMSlim model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of alpha for ADMMSlim model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("rho")
    @classmethod
    def check_rho(cls, v: list):
        """Validate rho."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of rho for ADMMSlim model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("it")
    @classmethod
    def check_it(cls, v: list):
        """Validate it."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    "Values of it for ADMMSlim model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("positive_only")
    @classmethod
    def check_positive_only(cls, v: list):
        """Validate positive_only."""
        v = convert_to_list(v)
        return v

    @field_validator("center_columns")
    @classmethod
    def check_center_columns(cls, v: list):
        """Validate center_columns."""
        v = convert_to_list(v)
        return v


@params_registry.register("VSM")
class VSM(RecomModel):
    """Definition of the model VSM.

    Attributes:
        similarity (STR_FIELD): List of names of similarity functions.
        user_profile (STR_FIELD): List of user profile computations.
        item_profile (STR_FIELD): List of item profile computations.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    similarity: STR_FIELD
    user_profile: STR_FIELD
    item_profile: STR_FIELD
    need_side_information: ClassVar[bool] = True

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        v = convert_to_list(v)
        for value in v:
            if not check_similarity(value):
                raise ValueError(
                    "Values of similarity for ItemKNN model must be supported similarities. "
                    f"Values received as input: {v}. "
                    f"Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("user_profile")
    @classmethod
    def check_user_profile(cls, v: list):
        """Validate user_profile."""
        v = convert_to_list(v)
        for value in v:
            if not check_user_profile(value):
                raise ValueError(
                    "Values of user_profile for VSM model must be 'binary' or 'tfidf'. "
                    f"Values received as input: {v}. "
                )
        return v

    @field_validator("item_profile")
    @classmethod
    def check_item_profile(cls, v: list):
        """Validate item_profile."""
        v = convert_to_list(v)
        for value in v:
            if not check_user_profile(value):
                raise ValueError(
                    "Values of item_profile for VSM model must be 'binary' or 'tfidf'. "
                    f"Values received as input: {v}. "
                )
        return v


@params_registry.register("CEASE")
class CEASE(RecomModel):
    """Definition of the model CEASE.

    Attributes:
        l2 (FLOAT_FIELD): List of values that l2 regularization can take.
        alpha (FLOAT_FIELD): List of values for alpha regularization.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    l2: FLOAT_FIELD
    alpha: FLOAT_FIELD
    need_side_information: ClassVar[bool] = True

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v: list):
        """Validate l2."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    f"Values of l2 for CEASE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of alpha for CEASE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("AddEASE")
class AddEASE(RecomModel):
    """Definition of the model AddEASE.

    Attributes:
        l2 (FLOAT_FIELD): List of values that l2 regularization can take.
        alpha (FLOAT_FIELD): List of values for alpha regularization.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    l2: FLOAT_FIELD
    alpha: FLOAT_FIELD
    need_side_information: ClassVar[bool] = True

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v: list):
        """Validate l2."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    f"Values of l2 for AddEASE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of alpha for AddEASE model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("AttributeItemKNN")
class AttributeItemKNN(RecomModel):
    """Definition of the model AttributeItemKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        normalize (BOOL_FIELD): List of values for normalization flag.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    normalize: BOOL_FIELD
    need_side_information: ClassVar[bool] = True

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of k for AttributeItemKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        v = convert_to_list(v)
        for value in v:
            if not check_similarity(value):
                raise ValueError(
                    "Values of similarity for AttributeItemKNN model must be supported similarities. "
                    f"Values received as input: {v}. "
                    f"Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        v = convert_to_list(v)
        return v


@params_registry.register("AttributeUserKNN")
class AttributeUserKNN(RecomModel):
    """Definition of the model AttributeUserKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        user_profile (STR_FIELD): List of user profile computations.
        normalize (BOOL_FIELD): List of values for normalization flag.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    user_profile: STR_FIELD
    normalize: BOOL_FIELD
    need_side_information: ClassVar[bool] = True

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of k for AttributeUserKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        v = convert_to_list(v)
        for value in v:
            if not check_similarity(value):
                raise ValueError(
                    "Values of similarity for AttributeUserKNN model must be supported similarities. "
                    f"Values received as input: {v}. "
                    f"Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("user_profile")
    @classmethod
    def check_user_profile(cls, v: list):
        """Validate user_profile."""
        v = convert_to_list(v)
        for value in v:
            if not check_user_profile(value):
                raise ValueError(
                    "Values of user_profile for AttributeUserKNN model must be 'binary' or 'tfidf'. "
                    f"Values received as input: {v}. "
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        v = convert_to_list(v)
        return v


@params_registry.register("Random")
class Random(RecomModel):
    """Empty definition of the model Random."""
