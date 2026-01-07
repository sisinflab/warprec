import os
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator
from warprec.utils.config.common import check_separator
from warprec.utils.enums import WritingMethods


class RecommendationWriting(BaseModel):
    """Definition of the recommendation sub-configuration part of the configuration file.

    Attributes:
        sep (str): The separator to use for the recommendation files.
        ext (str): The extension of the recommendation files.
        header (bool): Whether or not to write the header in the recommendation files.
        k (int): The number of recommendations to write in the recommendation files.
        user_label (str): The user label in the header of the file.
        item_label (str): The item label in the header of the file.
        rating_label (str): The rating label in the header of the file.
    """

    sep: str = "\t"
    ext: str = ".tsv"
    header: bool = True
    k: int = 50
    user_label: str = "user_id"
    item_label: str = "item_id"
    rating_label: str = "rating"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)

    def get_header(self) -> bool | List[str]:
        """Returns the header of the recommendation file."""
        if self.header:
            return [self.user_label, self.item_label, self.rating_label]
        return False


class ResultsWriting(BaseModel):
    """Definition of the results sub-configuration part of the configuration file.

    Attributes:
        sep (str): The separator to use for the results files.
        ext (str): The extension of the results files.
    """

    sep: str = "\t"
    ext: str = ".tsv"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)


class SplitWriting(BaseModel):
    """Definition of the split sub-configuration part of the configuration file.

    Attributes:
        sep (str): The separator to use for the split files.
        ext (str): The extension of the split files.
        header (bool): Whether or not to write the header in the split files.
    """

    sep: str = "\t"
    ext: str = ".tsv"
    header: bool = True

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
        azure_blob_experiment_container (Optional[str]): The Azure Blob container name.
        save_split (Optional[bool]): Whether or not to save the splits created for later use.
        results (ResultsWriting): The configuration of the results writing process.
        split (SplitWriting): The configuration of the split writing process.
        recommendation (RecommendationWriting): The configuration of the result writing process.
    """

    dataset_name: str
    writing_method: WritingMethods
    local_experiment_path: Optional[str] = None
    azure_blob_experiment_container: Optional[str] = None
    save_split: Optional[bool] = False
    results: ResultsWriting = Field(default_factory=ResultsWriting)
    split: SplitWriting = Field(default_factory=SplitWriting)
    recommendation: RecommendationWriting = Field(default_factory=RecommendationWriting)

    @model_validator(mode="after")
    def model_validation(self):
        if self.writing_method == WritingMethods.LOCAL:
            if not self.local_experiment_path:
                raise ValueError(
                    "When choosing local writing method a local path must be provided."
                )
            try:
                os.makedirs(self.local_experiment_path, exist_ok=True)
            except OSError as e:
                raise ValueError(
                    f"The local path provided {self.local_experiment_path} "
                    f"is not accessible or writable. Error: {e}"
                )
        return self
