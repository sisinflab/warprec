from typing import Optional

from pydantic import BaseModel, Field, field_validator
from warprec.utils.config.common import check_separator


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
        return check_separator(v)


class GeneralConfig(BaseModel):
    """Definition of the general configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        precision (Optional[str]): The precision to use during computation.
        batch_size (Optional[int]): The batch_size used during the experiment.
        ray_verbose (Optional[int]): The Ray level of verbosity.
        recommendation (Optional[GeneralRecommendation]): The general information
            about the recommendation.
    """

    precision: Optional[str] = "float32"
    batch_size: Optional[int] = 1024
    ray_verbose: Optional[int] = 1
    recommendation: Optional[GeneralRecommendation] = Field(
        default_factory=GeneralRecommendation
    )
