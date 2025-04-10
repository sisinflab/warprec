from pydantic import field_validator
from elliotwo.utils.config.model_configuration import (
    RecomModel,
    LIST_INT_FIELD,
    FLOAT_INT_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    BOOL_FIELD,
)
from elliotwo.utils.enums import SearchSpace
from elliotwo.utils.registry import params_registry, similarities_registry


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (FLOAT_INT_FIELD): List of values that l2 regularization can take.
    """

    l2: FLOAT_INT_FIELD

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v: list):
        """Validate l2."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, (float, int)) and value <= 0:
                raise ValueError(
                    f"Values of l2 for EASE model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (FLOAT_INT_FIELD): List of values that l1 regularization can take.
        alpha (FLOAT_INT_FIELD): List of values that alpha can take.
    """

    l1: FLOAT_INT_FIELD
    alpha: FLOAT_INT_FIELD

    @field_validator("l1")
    @classmethod
    def check_l1(cls, v: list):
        """Validate l1."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, (float, int)) and (value < 0 or value > 1):
                raise ValueError(
                    "Values of l1 for Slim model must be in [0, 1] range. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, (float, int)) and value < 0:
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
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, int) and value <= 0:
                raise ValueError(
                    "Values of k for ItemKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if (
                isinstance(value, str)
                and value.lower() != SearchSpace.CHOICE.value
                and value.lower() != SearchSpace.GRID.value
                and value.upper() not in similarities_registry.list_registered()
            ):
                raise ValueError(
                    "Values of similarity for ItemKNN model must be supported similarities. "
                    f"Values received as input: {v}. Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        if not isinstance(v, list):
            v = [v]
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
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, int) and value <= 0:
                raise ValueError(
                    "Values of k for UserKNN model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if (
                isinstance(v, str)
                and v.lower() != SearchSpace.CHOICE.value
                and v.lower() != SearchSpace.GRID.value
                and v.upper() not in similarities_registry.list_registered()
            ):
                raise ValueError(
                    "Values of similarity for UserKNN model must be supported similarities. "
                    f"Values received as input: {v}. Supported similarities: {similarities_registry.list_registered()}"
                )
        return v

    @field_validator("normalize")
    @classmethod
    def check_normalize(cls, v: list):
        """Validate normalize."""
        if not isinstance(v, list):
            v = [v]
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
    """

    mf_embedding_size: INT_FIELD
    mlp_embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    mf_train: BOOL_FIELD
    mlp_train: BOOL_FIELD
    dropout: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("mf_embedding_size")
    @classmethod
    def check_mf_embedding_size(cls, v: list):
        """Validate mf_embedding_size."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, int) and value <= 0:
                raise ValueError(
                    "Values of mf_embedding_size for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("mlp_embedding_size")
    @classmethod
    def check_mlp_embedding_size(cls, v: list):
        """Validate mlp_embedding_size."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, int) and value <= 0:
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
        if not isinstance(v, list):
            v = [v]
        return v

    @field_validator("mlp_train")
    @classmethod
    def check_mlp_train(cls, v: list):
        """Validate mlp_train."""
        if not isinstance(v, list):
            v = [v]
        return v

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, float) and value < 0:
                raise ValueError(
                    "Values of dropout for NeuMF model must be >= 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, int) and value <= 0:
                raise ValueError(
                    "Values of epochs for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        if not isinstance(v, list):
            v = [v]
        for value in v:
            if isinstance(value, float) and value <= 0:
                raise ValueError(
                    "Values of learning_rate for NeuMF model must be > 0. "
                    f"Values received as input: {v}"
                )
        return v
