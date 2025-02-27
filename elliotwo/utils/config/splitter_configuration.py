# pylint: disable=E1101
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.utils.logger import logger


class SplittingConfig(BaseModel):
    """Definition of the splitting configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        strategy (Optional[SplittingStrategies]): The splitting strategy to be used to split data.
        validation (Optional[bool]): Whether or not to create a validation
            split during the splitting process.
        ratio (Optional[List[float]]): The ratios of the splitting to be created.
        k (Optional[int]): The number of examples to remove during leave-k-out strategy.
        save_split (Optional[bool]): Whether or not to save the splits created for later use.
    """

    strategy: Optional[SplittingStrategies] = SplittingStrategies.NONE
    validation: Optional[bool] = False
    ratio: Optional[List[float]] = None
    k: Optional[int] = None
    save_split: Optional[bool] = False

    @field_validator("ratio")
    @classmethod
    def check_sum_to_one(cls, v):
        """Validates the ratios.

        Raise:
            ValueError: If the list length is different from 2 or 3.
            ValueError: The sum of the values isn't 1.
        """
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
            _expected_ratio = 3 if self.validation else 2
            if len(self.ratio) != _expected_ratio:
                raise ValueError(
                    "The ratio and the number of split expected "
                    "do not match. Check if validation set parameter "
                    "has been set or if ratio values are correct."
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

        return self
