from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from elliotwo.utils.enums import RatingType
from elliotwo.utils.logger import logger


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


class DataConfig(BaseModel):
    """Definition of the data configuration part of the configuration file.

    Attributes:
        dataset_name (str): Name of the dataset.
        loading_strategy (str): The strategy to use to load the data. Can be 'dataset' or 'split'.
        data_type (str): The type of data to be loaded. Can be 'transaction'.
        local_path (Optional[str]): Path to the file containing the transaction data.
        split_dir (Optional[str]): The directory where the splits are saved.
        experiment_path (Optional[str]): The local experiment path.
        sep (Optional[str]): Custom separator for the file containing the transaction data.
        rating_type (RatingType): The type of rating to be used. If 'implicit' is chosen,
            the reader will not look for a score.
        batch_size (Optional[int]): The batch size to be used during the reading process.
            If None is chosen, the data will be read in one pass.
        labels (Labels): The labels sub-configuration. Defaults to Labels default values.
        dtypes (CustomDtype): The list of column dtype.
    """

    dataset_name: str
    loading_strategy: str
    data_type: str
    local_path: Optional[str] = None
    split_dir: Optional[str] = None
    experiment_path: Optional[str] = "./experiments/"
    sep: Optional[str] = ","
    rating_type: RatingType
    batch_size: Optional[int] = 1024
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

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f"The string {v} is not a valid separator. Using default separator {','}."
            )
            v = ","
        return v

    @model_validator(mode="after")
    def check_data(self):
        """This method checks if the required information have been passed to the configuration."""
        # ValueError checks
        if self.loading_strategy == "split" and not self.split_dir:
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
        if self.loading_strategy == "dataset" and self.split_dir:
            logger.attention(
                "You have chosen dataset loading strategy but the split_dir field "
                "has been filled. Check your configuration file for possible errors."
            )
        return self
