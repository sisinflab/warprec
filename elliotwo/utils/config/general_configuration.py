# pylint: disable=E1101
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from elliotwo.utils.registry import metric_registry
from elliotwo.utils.logger import logger


class GeneralRecommendation(BaseModel):
    """Definition of recommendation information.

    Attributes:
        save_recs (Optional[bool]): Flag for recommendation saving. Defaults to False.
        sep (Optional[str]): Custom separator to use during recommendation saving. Defaults to ','.
        ext (Optional[str]): Custom extension. Defaults to '.csv'.
    """

    save_recs: Optional[bool] = False
    sep: Optional[str] = ","
    ext: Optional[str] = ".csv"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator.

        Raise:
            UnicodeDecodeError: If the separator is not correct.
        """
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
        """Validate device.

        Raise:
            ValueError: If the device is not in the correct format.
        """
        if v in ("cuda", "cpu"):
            return v
        if v.startswith("cuda:"):
            parts = v.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda[:index]".')

    @field_validator("validation_metric")
    @classmethod
    def check_validation_metric(cls, v: str):
        """Validate validation metric.

        Raise:
            ValueError: If the validation metric is not in the correct format.
        """
        if "@" not in v:
            raise ValueError(
                f"Validation metric {v} not valid. Validation metric "
                f"should be defined as: metric_name@top_k."
            )
        if v.count("@") > 1:
            raise ValueError(
                "Validation metric contains more than one @, check your configuration file."
            )
        metric, top_k = v.split("@")
        if metric not in metric_registry.list_registered():
            raise ValueError(
                f"Metric {metric} not in metric registry. This is the list"
                f"of supported metrics: {metric_registry.list_registered()}"
            )
        if not top_k.isnumeric():
            raise ValueError(
                "Validation metric should be provided with a top_k number."
            )
        return v

    @model_validator(mode="after")
    def model_validation(self):
        """This method validates the General configuration.

        Raise:
            ValueError: If some values are inconsistent in the configuration file.
        """
        if self.recommendation.save_recs and not self.setup_experiment:  # pylint: disable=no-member
            raise ValueError(
                "You are trying to save the recommendations without "
                "setting up the directory. Set setup_experiment to True."
            )
        return self
