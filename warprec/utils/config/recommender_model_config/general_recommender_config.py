from typing import ClassVar
from itertools import product

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
    check_between_zero_and_one,
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


@params_registry.register("Pop")
class Pop(RecomModel):
    """Empty definition of the model Pop."""


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
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of embedding_size for LightGCN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("n_layers")
    @classmethod
    def check_k(cls, v: list):
        """Validate n_layers."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of n_layers for LightGCN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_weight")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate reg_weight."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_weight for LightGCN model must be >= 0. "
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
                    "Values of epochs for LightGCN model must be > 0. "
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
                    "Values of learning_rate for LightGCN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v


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
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of embedding_size for LightGCN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_weight for LightGCN model must be >= 0. "
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
                    "Values of epochs for LightGCN model must be > 0. "
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
                    "Values of learning_rate for LightGCN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("weight_size")
    @classmethod
    def check_weight_size(cls, v: list):
        """Validate weight_size."""
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
        for weight_size in v:
            for value in weight_size:
                if value <= 0:
                    raise ValueError(
                        "Weight size for NGCF must be > 0. "
                        f"Values received as input: {v}"
                    )
        if strat:
            v.insert(0, strat)
        return v

    @field_validator("node_dropout")
    @classmethod
    def check_node_dropout(cls, v: list):
        """Validate node_dropout."""
        v = convert_to_list(v)
        for value in v:
            if check_between_zero_and_one(value):
                raise ValueError(
                    "Values of node_dropout for NGCF model must be >= 0 and <= 1. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("message_dropout")
    @classmethod
    def check_message_dropout(cls, v: list):
        """Validate message_dropout."""
        v = convert_to_list(v)
        for value in v:
            if check_between_zero_and_one(value):
                raise ValueError(
                    "Values of message_dropout for NGCF model must be >= 0 and <= 1. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("FISM")
class FISM(RecomModel):
    """Definition of the model FISM.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        reg_1 (FLOAT_FIELD): List of values for reg_1.
        reg_2 (FLOAT_FIELD): List of values for reg_2.
        alpha (FLOAT_FIELD): List of values for alpha.
        split_to (INT_FIELD): List of values for split_to.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    embedding_size: INT_FIELD
    reg_1: FLOAT_FIELD
    reg_2: FLOAT_FIELD
    alpha: FLOAT_FIELD
    split_to: INT_FIELD
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
                    "Values of embedding_size for FISM model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_1")
    @classmethod
    def check_reg_1(cls, v: list):
        """Validate reg_1."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_1 for FISM model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_2")
    @classmethod
    def check_reg_2(cls, v: list):
        """Validate reg_2."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_2 for FISM model must be >= 0. "
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
                    f"Values of alpha for FISM model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("split_to")
    @classmethod
    def check_split_to(cls, v: list):
        """Validate split_to."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of split_to for FISM model must be > 0. "
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
                    "Values of epochs for FISM model must be > 0. "
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
                    "Values of learning_rate for FISM model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("ConvNCF")
class ConvNCF(RecomModel):
    """Definition of the model ConvNCF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        cnn_channels (LIST_INT_FIELD): List of values for CNN channels.
        cnn_kernels (LIST_INT_FIELD): List of values for CNN kernels.
        cnn_strides (LIST_INT_FIELD): List of values for CNN strides.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        reg_embedding (FLOAT_FIELD): List of values for embedding regularization.
        reg_cnn_mlp (FLOAT_FIELD): List of values for CNN and MLP regularization.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        need_single_trial_validation (ClassVar[bool]): Whether or not to check if a Ray Tune
            trial parameter are valid.
    """

    embedding_size: INT_FIELD
    cnn_channels: LIST_INT_FIELD
    cnn_kernels: LIST_INT_FIELD
    cnn_strides: LIST_INT_FIELD
    dropout_prob: FLOAT_FIELD
    reg_embedding: FLOAT_FIELD
    reg_cnn_mlp: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        v = convert_to_list(v)
        for value in v:
            if check_less_equal_zero(value):
                raise ValueError(
                    "Values of embedding_size for ConvNCF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("cnn_channels")
    @classmethod
    def check_cnn_channels(cls, v: list):
        """Validate cnn_channels."""
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
        for weight_size in v:
            for value in weight_size:
                if value <= 0:
                    raise ValueError(
                        "cnn_channels for ConvNCF must be > 0. "
                        f"Values received as input: {v}"
                    )
        if strat:
            v.insert(0, strat)
        return v

    @field_validator("cnn_kernels")
    @classmethod
    def check_cnn_kernels(cls, v: list):
        """Validate cnn_kernels."""
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
        for weight_size in v:
            for value in weight_size:
                if value <= 0:
                    raise ValueError(
                        "cnn_kernels for ConvNCF must be > 0. "
                        f"Values received as input: {v}"
                    )
        if strat:
            v.insert(0, strat)
        return v

    @field_validator("cnn_strides")
    @classmethod
    def check_cnn_strides(cls, v: list):
        """Validate cnn_strides."""
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
        for weight_size in v:
            for value in weight_size:
                if value <= 0:
                    raise ValueError(
                        "cnn_strides for ConvNCF must be > 0. "
                        f"Values received as input: {v}"
                    )
        if strat:
            v.insert(0, strat)
        return v

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of dropout_prob for ConvNCF model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_embedding")
    @classmethod
    def check_reg_embedding(cls, v: list):
        """Validate reg_embedding."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_embedding for ConvNCF model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("reg_cnn_mlp")
    @classmethod
    def check_reg_cnn_mlp(cls, v: list):
        """Validate reg_cnn_mlp."""
        v = convert_to_list(v)
        for value in v:
            if check_less_than_zero(value):
                raise ValueError(
                    f"Values of reg_cnn_mlp for ConvNCF model must be >= 0. "
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
                    "Values of epochs for ConvNCF model must be > 0. "
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
                    "Values of learning_rate for ConvNCF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.
        This method should be called after all individual fields have been validated.

        Raises:
            ValueError: If no valid combination of hyperparameters can be formed.
        """
        # Extract parameters to check, removing searching strategy
        embedding_sizes = self._clean_param_list(self.embedding_size)
        cnn_channels_list = self._clean_param_list(self.cnn_channels)
        cnn_kernels_list = self._clean_param_list(self.cnn_kernels)
        cnn_strides_list = self._clean_param_list(self.cnn_strides)

        # Check if parameters are lists of lists
        cnn_channels_processed = [
            item if isinstance(item, list) else [item] for item in cnn_channels_list
        ]
        cnn_kernels_processed = [
            item if isinstance(item, list) else [item] for item in cnn_kernels_list
        ]
        cnn_strides_processed = [
            item if isinstance(item, list) else [item] for item in cnn_strides_list
        ]

        # Iter over all possible combinations and check if
        # any of them is valid.
        has_valid_combination = False
        for emb_size, channels_config, kernels_config, strides_config in product(
            embedding_sizes,
            cnn_channels_processed,
            cnn_kernels_processed,
            cnn_strides_processed,
        ):
            # Check for lengths
            if not (len(channels_config) == len(kernels_config) == len(strides_config)):
                continue
            # Check for embedding size
            if emb_size != channels_config[0]:
                continue

            # Found a valid combination
            has_valid_combination = True
            break

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for ConvNCF. "
                "Ensure there's at least one combination of 'embedding_size', "
                "'cnn_channels', 'cnn_kernels', and 'cnn_strides' that meets the criteria: "
                "1. The lengths of 'cnn_channels', 'cnn_kernels', and 'cnn_strides' must be equal. "
                "2. The dimension of the first CNN channel must be equal to 'embedding_size'."
            )

    def validate_single_trial_params(self):
        """Validates the coherence of cnn_channels, cnn_kernels, and cnn_strides
        for a single trial's parameter set.

        Raises:
            ValueError: If the parameter values are not consistent for the model.
        """
        # Clean parameters from search space information
        embedding_size_clean = (
            self.embedding_size[1:]
            if self.embedding_size and isinstance(self.embedding_size[0], str)
            else self.embedding_size
        )
        cnn_channels_clean = (
            self.cnn_channels[1:]
            if self.cnn_channels and isinstance(self.cnn_channels[0], str)
            else self.cnn_channels
        )
        cnn_kernels_clean = (
            self.cnn_kernels[1:]
            if self.cnn_kernels and isinstance(self.cnn_kernels[0], str)
            else self.cnn_kernels
        )
        cnn_strides_clean = (
            self.cnn_strides[1:]
            if self.cnn_strides and isinstance(self.cnn_strides[0], str)
            else self.cnn_strides
        )

        # Track the lengths of layers
        len_channels = len(cnn_channels_clean[0])
        len_kernels = len(cnn_kernels_clean[0])
        len_strides = len(cnn_strides_clean[0])

        # Check if this is a possible combination of parameters
        # if not, just raise an error.
        # RayTune will skip this trial
        if not (len_channels == len_kernels == len_strides):
            raise ValueError(
                f"Inconsistent CNN layer configuration: "
                f"cnn_channels length ({len_channels}), cnn_kernels length ({len_kernels}), "
                f"and cnn_strides length ({len_strides}) must be equal. "
            )

        emb_size = embedding_size_clean[0]
        first_layer_cnn = cnn_channels_clean[0][0]

        # Check if the first cnn_channel output layer
        # is the same as the embedding size
        if emb_size != first_layer_cnn:
            raise ValueError(
                f"Embedding size must be the same as the first layer of CNN. "
                f"embedding_size value ({emb_size}), first cnn layer ({first_layer_cnn}). "
            )
