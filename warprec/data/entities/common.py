from typing import Any, Optional

import narwhals as nw
import numpy as np
import torch
from narwhals.dataframe import DataFrame
from torch.utils.data import DataLoader, Dataset

USER_INDEX = "__uidx__"
ITEM_INDEX = "__iidx__"


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker so that sampling is reproducible.

    Each worker receives its own copy of the dataset, so without this they would
    all inherit the same generator state and draw the same negatives. Torch
    already gives every worker a distinct, seed-derived initial seed, so that is
    what the copies are reseeded from: the run stays reproducible while the
    workers stay independent of one another.

    Args:
        worker_id (int): The index of the worker being seeded.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    dataset = getattr(info, "dataset", None)
    for holder in (dataset, getattr(dataset, "sampler", None)):
        if hasattr(holder, "rng"):
            holder.rng = np.random.default_rng(worker_seed)


def seeded_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    **kwargs: Any,
) -> DataLoader:
    """Wrap a dataset in a DataLoader whose shuffling and sampling are reproducible.

    Every entity builds its loaders the same way, so the generator and the worker
    seeding live here rather than being repeated at each call site, where one copy
    could silently drift from the rest.

    Args:
        dataset (Dataset): The dataset to iterate.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.
        seed (int): The seed that makes an epoch reproducible.
        **kwargs (Any): Additional keyword arguments for the DataLoader.

    Returns:
        DataLoader: The seeded loader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def map_to_index_space(
    data: DataFrame[Any],
    user_label: str,
    item_label: str,
    user_mapping: dict,
    item_mapping: dict,
    keep: Optional[list] = None,
) -> DataFrame[Any]:
    """Attach the user and item indices to a frame of raw interactions.

    The two joins are inner ones on purpose: a row naming a user or an item the
    mappings do not know is outside the catalogue the experiment was built on and
    is dropped. That rule is the reason this lives in one place - the row-oriented
    entities each used to carry their own copy of it, and a copy is a chance for
    the two to disagree about which rows exist.

    Args:
        data (DataFrame[Any]): The raw interactions.
        user_label (str): The name of the user column.
        item_label (str): The name of the item column.
        user_mapping (dict): Mapping of user ID -> user index.
        item_mapping (dict): Mapping of item ID -> item index.
        keep (Optional[list]): Columns to carry through in addition to the
            indices. Left out, every column of the input is kept.

    Returns:
        DataFrame[Any]: The frame with the index columns attached.
    """
    namespace = nw.get_native_namespace(data)
    umap_df = nw.from_dict(
        {
            user_label: list(user_mapping.keys()),
            USER_INDEX: list(user_mapping.values()),
        },
        native_namespace=namespace,
    )
    imap_df = nw.from_dict(
        {
            item_label: list(item_mapping.keys()),
            ITEM_INDEX: list(item_mapping.values()),
        },
        native_namespace=namespace,
    )

    mapped = data.join(umap_df, on=user_label, how="inner").join(
        imap_df, on=item_label, how="inner"
    )

    if keep is None:
        return mapped
    return mapped.select([nw.col(c) for c in (*keep, USER_INDEX, ITEM_INDEX)])
