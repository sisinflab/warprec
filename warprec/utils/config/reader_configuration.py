from typing import Optional, List, ClassVar, Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from warprec.utils.enums import RatingType, ReadingMethods
from warprec.utils.config.common import check_separator, Labels
from warprec.utils.logger import logger


class SplitReading(BaseModel):
    """Definition of the split reading sub-configuration.

    This class reads all the information needed to load previously split data.

    Attributes:
        local_path (Optional[str]): The directory where the splits are saved.
        ext (Optional[str]): The extension of the split files.
        sep (Optional[str]): The separator of the split files.
        header (Optional[bool]): Whether the file has a header or not. Defaults to True.
    """

    local_path: Optional[str] = None
    ext: Optional[str] = ".tsv"
    sep: Optional[str] = "\t"
    header: Optional[bool] = True

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)


class SideInformationReading(BaseModel):
    """Definition of the side information reading sub-configuration.

    This class reads all the information needed to load side information data.

    Attributes:
        local_path (Optional[str]): The directory where the side information are saved.
        sep (Optional[str]): The separator of the split files.
        header (Optional[bool]): Whether the file has a header or not. Defaults to True.
    """

    local_path: Optional[str] = None
    sep: Optional[str] = "\t"
    header: Optional[bool] = True

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)


class ClusteringInformationReading(BaseModel):
    """Definition of the clustering information reading sub-configuration.

    This class reads all the information needed to load clustering information data.

    Attributes:
        user_local_path (Optional[str]): The path to the user clustering information.
        item_local_path (Optional[str]): The path to the item clustering information.
        user_sep (Optional[str]): The separator of the user clustering file.
        item_sep (Optional[str]): The separator of the item clustering file.
        user_header (Optional[bool]): Whether the user clustering file has a header. Defaults to True.
        item_header (Optional[bool]): Whether the item clustering file has a header. Defaults to True.
    """

    user_local_path: Optional[str] = None
    item_local_path: Optional[str] = None
    user_sep: Optional[str] = "\t"
    item_sep: Optional[str] = "\t"
    user_header: Optional[bool] = True
    item_header: Optional[bool] = True

    @field_validator("user_sep", "item_sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator."""
        return check_separator(v)


class CustomDtype(BaseModel):
    """Definition of the custom dtype sub-configuration.

    This class reads and optionally overrides default labels of important data.

    Attributes:
        user_id_type (Optional[str]): The dtype to format the user_id column.
        item_id_type (Optional[str]): The dtype to format the item_id column.
        rating_type (Optional[str]): The dtype to format the rating column.
        timestamp_type (Optional[str]): The dtype to format the timestamp column.
        cluster_type (Optional[str]): The dtype to format the cluster column.
    """

    user_id_type: Optional[str] = "int32"
    item_id_type: Optional[str] = "int32"
    rating_type: Optional[str] = "float32"
    timestamp_type: Optional[str] = "int32"
    cluster_type: Optional[str] = "int32"


class ReaderConfig(BaseModel):
    """Definition of the reader configuration part of the configuration file.

    Attributes:
        loading_strategy (str): The strategy to use to load the data. Can be 'dataset' or 'split'.
        data_type (str): The type of data to be loaded. Can be 'transaction'.
        reading_method (ReadingMethods): The strategy used to read the data.
        local_path (Optional[str | None]): The path to the local dataset.
        sep (Optional[str]): The separator of the file to read.
        header (Optional[bool]): Whether the file has a header or not. Defaults to True.
        rating_type (RatingType): The type of rating to be used. If 'implicit' is chosen,
            the reader will not look for a score.
        split (Optional[SplitReading]): The information of the split reading process.
        side (Optional[SideInformationReading]): The side information of the dataset.
        clustering (Optional[ClusteringInformationReading]): The clustering information
            of the dataset.
        labels (Labels): The labels sub-configuration. Defaults to Labels default values.
        dtypes (CustomDtype): The list of column dtype.
        column_map_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy counterpart.
    """

    loading_strategy: str
    data_type: str
    reading_method: ReadingMethods
    local_path: Optional[str | None] = None
    sep: Optional[str] = "\t"
    header: Optional[bool] = True
    rating_type: RatingType
    split: Optional[SplitReading] = Field(default_factory=SplitReading)
    side: Optional[SideInformationReading] = None
    clustering: Optional[ClusteringInformationReading] = None
    labels: Labels = Field(default_factory=Labels)
    dtypes: CustomDtype = Field(default_factory=CustomDtype)

    # Supported dtype
    column_map_dtype: ClassVar[dict] = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32,
        "float64": np.float64,
        "str": np.str_,
    }

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
        return check_separator(v)

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

        # Final checks and parsing
        self.check_column_dtype()

        return self

    def column_names(self) -> List[str]:
        """This method returns the names of the column passed through configuration.

        Returns:
            List[str]: The list of column names.
        """
        return [
            self.labels.user_id_label,
            self.labels.item_id_label,
            self.labels.rating_label,
            self.labels.timestamp_label,
        ]

    def column_dtype(self) -> Dict[str, np.dtype]:
        """This method will parse the dtype from the string forma to their numpy counterpart.

        Returns:
            Dict[str, np.dtype]: A list containing the dtype to use for data loading.
        """
        column_names = self.column_names()
        column_dtypes = [
            self.dtypes.user_id_type,
            self.dtypes.item_id_type,
            self.dtypes.rating_type,
            self.dtypes.timestamp_type,
        ]
        return {
            name: self.column_map_dtype[dtype]
            for name, dtype in zip(column_names, column_dtypes)
        }

    def check_column_dtype(self) -> None:
        """This method validates the custom dtype passed with the configuration file.

        Raises:
            ValueError: If the dtype are not supported or incorrect.
        """
        for dtype_str in self.dtypes.model_dump().values():
            if dtype_str not in self.column_map_dtype:
                raise ValueError(
                    f"Custom dtype {dtype_str} not supported as a column data type."
                )
