from typing import List, Tuple, Optional
from pathlib import Path
from io import StringIO

import pandas as pd
import joblib
from pandas import DataFrame

from warprec.data.reader.base_reader import Reader


class LocalReader(Reader):
    """This class extends Reader and handles data reading from a local machine."""

    def read_tabular(
        self,
        local_path: str,
        column_names: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        sep: str = "\t",
        header: bool = True,
    ) -> DataFrame:
        """Reads tabular data (e.g., CSV, TSV) from a local file.

        The file content is read into memory and then processed robustly by the
        parent's stream processor.

        Args:
            local_path (str): The local file path to the tabular data.
            column_names (Optional[List[str]]): A list of expected column names.
            dtypes (Optional[List[str]]): A list of data types corresponding to `column_names`.
            sep (str): The delimiter character used in the file. Defaults to tab `\t`.
            header (bool): A boolean indicating if the file has a header row. Defaults to `True`.

        Returns:
            DataFrame: A pandas DataFrame containing the tabular data. Returns an empty DataFrame
                if the blob is not found.
        """
        path = Path(local_path)
        if not path.exists():
            # Return an empty df that the split logic can check
            return pd.DataFrame()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        stream = StringIO(content)

        # Create mapping only if values are not None
        dtypes_map = None
        if column_names and dtypes:
            dtypes_map = dict(zip(column_names, dtypes))

        return self._process_csv_stream(
            stream=stream,
            sep=sep,
            header=header,
            desired_cols=column_names,
            desired_dtypes=dtypes_map,
        )

    def read_tabular_split(  # type: ignore[override]
        self,
        split_dir: str,
        column_names: Optional[List[str]],
        dtypes: Optional[List[str]],
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        return super().read_tabular_split(
            base_location=split_dir,
            column_names=column_names,
            dtypes=dtypes,
            sep=sep,
            ext=ext,
            header=header,
            is_remote=False,  # Specify local path handling
        )

    def read_json(self, *args, **kwargs):
        """This method will read the json data from the source."""
        raise NotImplementedError

    def read_json_split(self, *args, **kwargs):
        """This method will read the json split data from the source."""
        raise NotImplementedError

    def load_model_state(self, local_path: str) -> dict:
        """Loads a model state from a given path.

        Args:
            local_path (str): The path to the model state file.

        Returns:
            dict: The deserialized information of the model (e.g., weights, hyperparameters)
                loaded using `joblib`.

        Raises:
            FileNotFoundError: If the model state was not found in the provided path.
        """
        path = Path(local_path)
        if path.exists():
            return joblib.load(path)
        raise FileNotFoundError(f"Model state not found in {path}")
