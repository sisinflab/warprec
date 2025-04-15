from typing import Optional, Union

from pydantic import BaseModel, model_validator, field_validator
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.utils.logger import logger


class SplittingConfig(BaseModel):
    """Definition of the splitting configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        strategy (Optional[SplittingStrategies]): The splitting strategy to be used to split data.
        test_ratio (Optional[float]): The ratio of the test set.
        val_ratio (Optional[float]): The ratio of the val set.
        test_k (Optional[int]): The k value for the test set.
        val_k (Optional[int]): The k value for the val set.
        timestamp (Optional[Union[int, str]]): The timestamp to be used for the test set.
            Either and integer or 'best'.
        seed (Optional[int]): The seed to be used during the splitting process.
    """

    strategy: Optional[SplittingStrategies] = SplittingStrategies.NONE
    test_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_k: Optional[int] = None
    val_k: Optional[int] = None
    timestamp: Optional[Union[int, str]] = None
    seed: Optional[int] = 42

    @field_validator("timestamp")
    @classmethod
    def check_timestamp(cls, v: Optional[Union[int, str]]):
        if v and isinstance(v, str):
            if v != "best":
                raise ValueError(
                    f"Timestamp must be either an integer or 'best'. You passed {v}."
                )
        return v

    @model_validator(mode="after")
    def check_dependencies(self):
        """This method checks if the required information have been passed to the configuration.

        Raise:
            ValueError: If an important field has not been filled with the correct information.
            Warning: If a field that will not be used during the experiment has been filled.
        """
        tol = 1e-6  # Tolerance will be used to check ratios

        # ValueError checks
        if self.strategy in [
            SplittingStrategies.RANDOM,
            SplittingStrategies.TEMPORAL_HOLDOUT,
        ]:
            if self.test_ratio is None:
                raise ValueError(
                    f"You have chosen {self.strategy.value} splitting but "
                    "the test ratio field has not been filled."
                )

        if self.test_ratio and self.val_ratio:
            if self.test_ratio + self.val_ratio + tol >= 1.0:
                raise ValueError(
                    "The test and validation ratios are too high and "
                    "there is no space for train set. Check you values."
                )

        if (
            self.strategy == SplittingStrategies.TEMPORAL_LEAVE_K_OUT
            and self.test_k is None
        ):
            raise ValueError(
                "You have chosen temporal leave k out splitting but "
                "the test k field has not been filled."
            )

        if (
            self.strategy == SplittingStrategies.TIMESTAMP_SLICING
            and self.timestamp is None
        ):
            raise ValueError(
                "You have chosen fixed timestamp splitting but "
                "the test timestamp field has not been filled."
            )

        # Attention checks
        if self.strategy in [
            SplittingStrategies.RANDOM,
            SplittingStrategies.TEMPORAL_HOLDOUT,
        ] and (self.test_k or self.val_k):
            logger.attention(
                f"You have filled the k field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )
        if self.strategy == SplittingStrategies.LEAVE_ONE_OUT and (
            self.test_ratio or self.val_ratio
        ):
            logger.attention(
                "You have filled the ratio field but splitting strategy "
                "has been set to leave-one-out. Check your "
                "configuration file for possible errors."
            )

        return self
