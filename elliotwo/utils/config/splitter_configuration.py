from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.utils.logger import logger


class SplittingConfig(BaseModel):
    """Definition of the splitting configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        strategy (Optional[SplittingStrategies]): The splitting strategy to be used to split data.
        ratio (Optional[List[float]]): The ratios of the splitting to be created.
        k (Optional[List[int]]): The number of examples to remove during leave-k-out strategy.
        validation (Optional[bool]): Whether or not to create a validation set.
        seed (Optional[int]): The seed to be used during the splitting process.
    """

    strategy: Optional[SplittingStrategies] = SplittingStrategies.NONE
    ratio: Optional[List[float]] = None
    k: Optional[List[int]] = None
    validation: Optional[bool] = False
    seed: Optional[int] = 42

    @field_validator("ratio")
    @classmethod
    def check_sum_to_one(cls, v: list):
        """Validates the ratios."""
        tol = 1e-6
        if v is not None:
            if len(v) not in [2, 3]:
                raise ValueError(
                    "List must be of length 2 in case of train/test split or 3 "
                    "in case of train/val/test split."
                )
            if not abs(sum(v) - 1.0) < tol:  # Slight tolerance for the sum
                raise ValueError(
                    f"The sum of ratios must be 1. Received sum: {sum(v)}. "
                    f"Accepted tolerance: {tol}"
                )
            if len(v) == 2:
                v.append(None)  # Add None for validation set
        return v

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validates the k values."""
        if v is not None:
            if len(v) not in [1, 2]:
                raise ValueError(
                    "List must be of length 1 in case of train/test split or 2 "
                    "in case of train/val/test split."
                )
            if len(v) == 1:
                v.append(None)  # Add None for validation set
        return v

    @model_validator(mode="after")
    def check_dependencies(self):
        """This method checks if the required information have been passed to the configuration.

        Raise:
            ValueError: If an important field has not been filled with the correct information.
            Warning: If a field that will not be used during the experiment has been filled.
        """
        # ValueError checks
        if self.strategy in [SplittingStrategies.RANDOM, SplittingStrategies.TEMPORAL]:
            if self.ratio is None:
                raise ValueError(
                    f"You have chosen {self.strategy.value} splitting but "
                    "the ratio field has not been filled."
                )

        # Attention checks
        if (
            self.strategy in [SplittingStrategies.RANDOM, SplittingStrategies.TEMPORAL]
            and self.k
        ):
            logger.attention(
                f"You have filled the k field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )
        if self.strategy == SplittingStrategies.LEAVE_ONE_OUT and self.ratio:
            logger.attention(
                "You have filled the ratio field but splitting strategy "
                "has been set to leave-one-out. Check your "
                "configuration file for possible errors."
            )
        if self.validation and self.ratio[2] is None:  # pylint: disable=unsubscriptable-object
            logger.attention(
                "You have chosen to create a validation set but the ratio "
                "field has not been filled with the correct information."
            )

        return self
