from typing import Union, List

from pydantic import field_validator
from elliotwo.utils.config.model_configuration import RecomModel
from elliotwo.utils.registry import params_registry


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (Union[List[Union[str, float, int]], float, int]):
            List of values that l2 regularization can take.
    """

    l2: Union[List[Union[str, float, int]], float, int]

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v: list):
        """Validate l2."""
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
        l1 (Union[List[Union[str, float, int]], float, int]): List of values
            that l1 regularization can take.
        alpha (Union[List[Union[str, float, int]], float, int]): List of values that alpha can take.
    """

    l1: Union[List[Union[str, float, int]], float, int]
    alpha: Union[List[Union[str, float, int]], float, int]

    @field_validator("l1")
    @classmethod
    def check_l1(cls, v: list):
        """Validate l1."""
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
        """Validate alpha"""
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
        k (Union[List[Union[str, int]], int]): List of values for neighbor.
        similarity (Union[List[str], str]): List of names of similarity functions.
        normalize (Union[List[Union[str, bool]], bool]): List of values for normalization flag.
    """

    k: Union[List[Union[str, int]], int]
    similarity: Union[List[str], str]
    normalize: Union[List[Union[str, bool]], bool]

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """TODO: Continue"""


@params_registry.register("UserKNN")
class UserKNN(RecomModel):
    """Definition of the model UserKNN.

    Attributes:
        k (Union[List[Union[str, int]], int]): List of values for neighbor.
        similarity (Union[List[str], str]): List of names of similarity functions.
        normalize (Union[List[Union[str, bool]], bool]): List of values for normalization flag.
    """

    k: Union[List[Union[str, int]], int]
    similarity: Union[List[str], str]
    normalize: Union[List[Union[str, bool]], bool]


@params_registry.register("NeuMF")
class NeuMF(RecomModel):
    """Definition of the model NeuMF.

    Attributes:
        mf_embedding_size (Union[List[Union[str, int]], int]): List of mf embedding size.
        mlp_embedding_size (Union[List[Union[str, int]], int]): List of mlp embedding size.
        mlp_hidden_size (Union[List[Union[str, List[int]]], List[List[int]], List[int]]):
            List of mlp_hidden_size values.
        mf_train (Union[List[Union[str, bool]], bool]): List of values for mf_train flag.
        mlp_train (Union[List[Union[str, bool]], bool]): List of values for mlp_train flag.
        dropout (Union[List[Union[str, float, int]], float, int]): List of values for dropout.
        epochs (Union[List[Union[str, int]], int]): List of values for epochs.
        learning_rate (Union[List[Union[str, float, int]], float, int]):
            List of values for learning rate.
    """

    mf_embedding_size: Union[List[Union[str, int]], int]
    mlp_embedding_size: Union[List[Union[str, int]], int]
    mlp_hidden_size: Union[List[Union[str, List[int]]], List[List[int]], List[int]]
    mf_train: Union[List[Union[str, bool]], bool]
    mlp_train: Union[List[Union[str, bool]], bool]
    dropout: Union[List[Union[str, float, int]], float, int]
    epochs: Union[List[Union[str, int]], int]
    learning_rate: Union[List[Union[str, float, int]], float, int]
