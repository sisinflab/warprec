from typing import Optional, Union

from pydantic import BaseModel, model_validator, field_validator, Field
from warprec.utils.enums import SplittingStrategies
from warprec.utils.logger import logger


class SplitStrategy(BaseModel):
    """Definition of a split criteria.

    Attributes:
        strategy (Optional[SplittingStrategies]): The splitting strategy to be used to split data.
        ratio (Optional[float]): The ratio value to pass to the splitting strategy.
        k (Optional[int]): The k value to pass to the splitting strategy.
        folds (Optional[int]): The folds value to pass to the splitting strategy.
        timestamp (Optional[Union[int, str]]): The timestamp to be used for the test set.
            Either and integer or 'best'.
        seed (Optional[int]): The seed to be used during the splitting process.
    """

    strategy: Optional[SplittingStrategies] = None
    ratio: Optional[float] = None
    k: Optional[int] = None
    folds: Optional[int] = None
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
        # ValueError checks
        if (
            self.strategy == SplittingStrategies.TEMPORAL_HOLDOUT
            or self.strategy == SplittingStrategies.RANDOM_HOLDOUT
        ) and self.ratio is None:
            raise ValueError(
                f"You have chosen {self.strategy.value} splitting but "
                "the ratio field has not been filled."
            )

        if (
            self.strategy == SplittingStrategies.TEMPORAL_LEAVE_K_OUT
            or self.strategy == SplittingStrategies.RANDOM_LEAVE_K_OUT
        ) and self.k is None:
            raise ValueError(
                f"You have chosen {self.strategy.value} splitting but "
                "the k field has not been filled."
            )

        if (
            self.strategy == SplittingStrategies.TIMESTAMP_SLICING
            and self.timestamp is None
        ):
            raise ValueError(
                "You have chosen fixed timestamp splitting but "
                "the timestamp field has not been filled."
            )

        if (
            self.strategy == SplittingStrategies.K_FOLD_CROSS_VALIDATION
            and self.folds is None
        ):
            raise ValueError(
                "You have chosen k-fold-cross-validation splitting but "
                "the fold field has not been filled."
            )

        # Attention checks
        if (
            self.strategy
            in [
                SplittingStrategies.TEMPORAL_HOLDOUT,
                SplittingStrategies.TIMESTAMP_SLICING,
                SplittingStrategies.RANDOM_HOLDOUT,
                SplittingStrategies.K_FOLD_CROSS_VALIDATION,
            ]
            and self.k
        ):
            logger.attention(
                f"You have filled the k field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )

        if (
            self.strategy
            in [
                SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
                SplittingStrategies.RANDOM_LEAVE_K_OUT,
                SplittingStrategies.K_FOLD_CROSS_VALIDATION,
            ]
            and self.ratio
        ):
            logger.attention(
                f"You have filled the ratio field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )

        if (
            self.strategy
            in [
                SplittingStrategies.TEMPORAL_HOLDOUT,
                SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
                SplittingStrategies.RANDOM_HOLDOUT,
                SplittingStrategies.RANDOM_LEAVE_K_OUT,
                SplittingStrategies.K_FOLD_CROSS_VALIDATION,
            ]
            and self.timestamp
        ):
            logger.attention(
                f"You have filled the timestamp field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )

        if (
            self.strategy
            in [
                SplittingStrategies.TEMPORAL_HOLDOUT,
                SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
                SplittingStrategies.TIMESTAMP_SLICING,
                SplittingStrategies.RANDOM_HOLDOUT,
                SplittingStrategies.RANDOM_LEAVE_K_OUT,
            ]
            and self.folds
        ):
            logger.attention(
                f"You have filled the folds field but the splitting strategy "
                f"has been set to {self.strategy.value}. Check your "
                "configuration file for possible errors."
            )

        return self


class SplittingConfig(BaseModel):
    """Definition of the splitting configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        test_splitting (Optional[SplitStrategy]): The criteria used to split the test set.
        validation_splitting (Optional[SplitStrategy]): The criteria used to split the validation set.
    """

    test_splitting: Optional[SplitStrategy] = Field(default_factory=SplitStrategy)
    validation_splitting: Optional[SplitStrategy] = Field(default_factory=SplitStrategy)

    @model_validator(mode="after")
    def check_dependencies(self) -> "SplittingConfig":
        """This method checks if the required information have been passed to the configuration.

        Returns:
            SplittingConfig: The validated configuration.

        Raises:
            ValueError: If the ratio values for test and validation set are too high.
        """
        tol = 1e-6  # Tolerance will be used to check ratios

        if self.test_splitting.ratio and self.validation_splitting.ratio:
            if self.test_splitting.ratio + self.validation_splitting.ratio + tol >= 1.0:
                raise ValueError(
                    "The test and validation ratios are too high and "
                    "there is no space for train set. Check you values."
                )

        if self.test_splitting.strategy == SplittingStrategies.K_FOLD_CROSS_VALIDATION:
            raise ValueError(
                "The test set cannot be created with k-fold-cross-validation. "
                "Choose another strategy"
            )

        return self
