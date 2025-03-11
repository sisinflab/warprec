from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from elliotwo.utils.logger import logger


class GeneralRecommendation(BaseModel):
    """Definition of recommendation information.

    Attributes:
        save_recs (Optional[bool]): Flag for recommendation saving. Defaults to False.
        sep (Optional[str]): Custom separator to use during recommendation saving. Defaults to ','.
        ext (Optional[str]): Custom extension. Defaults to '.csv'.
        k (Optional[int]): The number of recommendation per user.
    """

    save_recs: Optional[bool] = False
    sep: Optional[str] = ","
    ext: Optional[str] = ".csv"
    k: Optional[int] = 50

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f'The string {v} is not a valid separator. Using default separator ",".'
            )
            v = ","
        return v


class GeneralConfig(BaseModel):
    """Definition of the general configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        seed (Optional[int]): The seed that will be used during the experiment for reproducibility.
        float_digits (Optional[int]): The number of floating point digits to show on console.
        device (Optional[str]): The device that will be used for most operations.
        validation_metric (Optional[str]): The validation metric to use,
            in the format of metric_name@top_k.
        precision (Optional[str]): The precision to use during computation.
        max_eval (Optional[int]): The maximum number of evaluations to compute with hyperopt.
        recommendation (Optional[GeneralRecommendation]): The general information
            about the recommendation.
        setup_experiment (Optional[bool]): Wether or not to setup the experiment ambient.
    """

    seed: Optional[int] = 42
    float_digits: Optional[int] = 16
    device: Optional[str] = "cpu"
    validation_metric: Optional[str] = "nDCG@5"
    precision: Optional[str] = "float32"
    max_eval: Optional[int] = 10
    recommendation: Optional[GeneralRecommendation] = Field(
        default_factory=GeneralRecommendation
    )
    setup_experiment: Optional[bool] = True

    @field_validator("device")
    @classmethod
    def check_device(cls, v: str):
        """Validate device."""
        if v in ("cuda", "cpu"):
            return v
        if v.startswith("cuda:"):
            parts = v.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda[:index]".')

    @model_validator(mode="after")
    def model_validation(self):
        """This method validates the General configuration."""
        if self.recommendation.save_recs and not self.setup_experiment:
            raise ValueError(
                "You are trying to save the recommendations without "
                "setting up the directory. Set setup_experiment to True."
            )
        return self
