from typing import Dict

import pandas as pd

from warprec.utils.config.serving_configuration import ItemMetadata


def read_item_names(metadata: ItemMetadata) -> Dict[str, str]:
    """Item names by item id, read from a delimited file.

    Ids are kept as strings, which is how requests are matched against the
    model's mapping whatever type the dataset used for them.

    Args:
        metadata (ItemMetadata): Where the file is and how it is laid out.

    Returns:
        Dict[str, str]: The name of every item, by its id.
    """
    frame = pd.read_csv(
        metadata.path,
        sep=metadata.sep,
        header=0 if metadata.header else None,
        dtype=str,
        encoding=metadata.encoding,
        # Multi-character separators such as '::' need the python engine.
        engine="python" if len(metadata.sep) > 1 else "c",
        keep_default_na=False,
    )

    def column(key):
        return frame.iloc[:, key] if isinstance(key, int) else frame[key]

    return dict(zip(column(metadata.id_column), column(metadata.name_column)))
