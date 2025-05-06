import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from warprec.utils.config.common import check_separator
from warprec.utils.enums import WritingMethods


class WritingParams(BaseModel):
    """Definition of the writing params sub-configuration part of the configuration file.

    Attributes:
        sep (str): The separator to use for the recommendation files.
        ext (str): The extension of the recommendation files.
        user_label (str): The user label in the header of the file.
        item_label (str): The item label in the header of the file.
    """

    sep: str = "\t"
    ext: str = ".tsv"
    user_label: str = "user_id"
    item_label: str = "item_id"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)


class WriterConfig(BaseModel):
    """Definition of the writer configuration part of the configuration file.

    Attributes:
        dataset_name (str): Name of the dataset.
        writing_method (WritingMethods): The writing method that will be used.
        local_experiment_path (Optional[str]): Path to the file containing the transaction data.
        setup_experiment (bool): Flag value for the setup of the experiment.
        save_split (Optional[bool]): Whether or not to save the splits created for later use.
        writing_params (WritingParams): The configuration of the result writing process.
    """

    dataset_name: str
    writing_method: WritingMethods
    local_experiment_path: Optional[str] = None
    setup_experiment: bool = True
    save_split: Optional[bool] = False
    writing_params: WritingParams = Field(default_factory=WritingParams)

    @model_validator(mode="after")
    def model_validation(self):
        if self.writing_method == WritingMethods.LOCAL:
            if not self.local_experiment_path:
                raise ValueError(
                    "When choosing local writing method a local path must be provided."
                )
            if not os.path.exists(self.local_experiment_path):
                raise FileNotFoundError(
                    f"The local path provided {self.local_experiment_path} "
                    f"does not exists."
                )
        return self
