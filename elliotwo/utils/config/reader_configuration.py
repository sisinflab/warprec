from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from elliotwo.utils.enums import RatingType, ReadingMethods
from elliotwo.utils.logger import logger


class SplitReading(BaseModel):
    """Definition of the split reading sub-configuration.

    This class reads all the information needed to load previously split data.

    Attributes:
        local_path (Optional[str | None]): The directory where the splits are saved.
        ext (Optional[str]): The extension of the split files.
        sep (Optional[str]): The separator of the split files.
    """

    local_path: Optional[str | None] = None
    ext: Optional[str] = ".tsv"
    sep: Optional[str] = "\t"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f"The string {v} is not a valid separator. Using default separator {'\t'}."
            )
            v = "\t"
        return v


class ReadingParams(BaseModel):
    """Definition of the reading params sub-configuration.

    This class reads all the information needed to read the data correctly.

    Attributes:
        ext (Optional[str]): The extension of the file to read.
        sep (Optional[str]): The separator of the file to read.
        batch_size (Optional[int]): The batch size used during the reading process.
    """

    ext: Optional[str] = ".tsv"
    sep: Optional[str] = "\t"
    batch_size: Optional[int] = 1024

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f"The string {v} is not a valid separator. Using default separator {'\t'}."
            )
            v = "\t"
        return v


class Labels(BaseModel):
    """Definition of the label sub-configuration.

    This class reads and optionally overrides the default labels of important data.

    Attributes:
        user_id_label (Optional[str]): Name of the user ID label. Defaults to 'user_id'.
        item_id_label (Optional[str]): Name of the item ID label. Defaults to 'item_id'.
        rating_label (Optional[str]): Name of the rating label. Defaults to 'rating'.
        timestamp_label (Optional[str]): Name of the timestamp label. Defaults to 'timestamp'.
    """

    user_id_label: Optional[str] = "user_id"
    item_id_label: Optional[str] = "item_id"
    rating_label: Optional[str] = "rating"
    timestamp_label: Optional[str] = "timestamp"


class CustomDtype(BaseModel):
    """Definition of the custom dtype sub-configuration.

    This class reads and optionally overrides default labels of important data.

    Attributes:
        user_id_type (Optional[str]): The dtype to format the user_id column.
        item_id_type (Optional[str]): The dtype to format the item_id column.
        rating_type (Optional[str]): The dtype to format the rating column.
        timestamp_type (Optional[str]): The dtype to format the timestamp column.
    """

    user_id_type: Optional[str] = "int32"
    item_id_type: Optional[str] = "int32"
    rating_type: Optional[str] = "float32"
    timestamp_type: Optional[str] = "int32"


class ReaderConfig(BaseModel):
    """Definition of the reader configuration part of the configuration file.

    Attributes:
        loading_strategy (str): The strategy to use to load the data. Can be 'dataset' or 'split'.
        data_type (str): The type of data to be loaded. Can be 'transaction'.
        reading_method (ReadingMethods): The strategy used to read the data.
        local_path (Optional[str]): The path to the local dataset.
        rating_type (RatingType): The type of rating to be used. If 'implicit' is chosen,
            the reader will not look for a score.
        reading_params (Optional[ReadingParams]): The parameters of the reading process.
        split (Optional[SplitReading]): The information of the split reading process.
        labels (Labels): The labels sub-configuration. Defaults to Labels default values.
        dtypes (CustomDtype): The list of column dtype.
    """

    loading_strategy: str
    data_type: str
    reading_method: ReadingMethods
    local_path: Optional[str] = None
    rating_type: RatingType
    reading_params: Optional[ReadingParams] = Field(default_factory=ReadingParams)
    split: Optional[SplitReading] = Field(default_factory=SplitReading)
    labels: Labels = Field(default_factory=Labels)
    dtypes: CustomDtype = Field(default_factory=CustomDtype)

    @field_validator("loading_strategy")
    @classmethod
    def check_loading_strategy(cls, v: str):
        """Validates the loading strategy."""
        supported_strategies = ["dataset", "split"]
        if v not in supported_strategies:
            raise ValueError(
                f"Loading strategy {v} not supported. Supported strategies: {supported_strategies}."
            )
        return v

    @field_validator("data_type")
    @classmethod
    def check_data_type(cls, v: str):
        """Validates the data type."""
        supported_data_types = ["transaction"]
        if v not in supported_data_types:
            raise ValueError(
                f"Data type {v} not supported. Supported data types: {supported_data_types}."
            )
        return v

    @model_validator(mode="after")
    def check_data(self):
        """This method checks if the required information have been passed to the configuration."""
        # ValueError checks
        if self.loading_strategy == "split" and not self.split.local_path:
            raise ValueError(
                "You have chosen split loading strategy but the split_dir "
                "field has not been filled."
            )
        if self.loading_strategy == "dataset" and not self.local_path:
            raise ValueError(
                "You have chosen dataset loading strategy but the local_path "
                "field has not been filled."
            )

        # Attention checks
        if self.loading_strategy == "split" and self.local_path:
            logger.attention(
                "You have chosen split loading strategy but the local_path field "
                "has been filled. Check your configuration file for possible errors."
            )
        if self.loading_strategy == "dataset" and self.split.local_path:
            logger.attention(
                "You have chosen dataset loading strategy but the split_dir field "
                "has been filled. Check your configuration file for possible errors."
            )
        return self
