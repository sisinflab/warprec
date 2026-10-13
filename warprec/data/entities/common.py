import pickle  # nosec B403 - only WarpRec's own worker and checkpoint state
from typing import Any, Callable, Dict, List, Optional

import narwhals as nw
import numpy as np
import torch
from narwhals.dataframe import DataFrame
from torch.utils.data import DataLoader, Dataset, default_collate

USER_INDEX = "__uidx__"
ITEM_INDEX = "__iidx__"
ROW_INDEX = "__row__"


def _rng_holders(dataset: Any) -> List[Any]:
    """The objects a training dataset draws its random samples from.

    A dataset samples either through an 'rng' of its own or through the 'rng'
    of its negative sampler. Seeding a worker and saving a loader's state for a
    resume both have to reach the same generators, so they find them here.

    Args:
        dataset (Any): The dataset of a DataLoader.

    Returns:
        List[Any]: The dataset and its sampler, those of them holding an 'rng'.
    """
    return [
        holder
        for holder in (dataset, getattr(dataset, "sampler", None))
        if hasattr(holder, "rng")
    ]


def get_generator_state(generator: Any) -> Any:
    """Read the state of a random generator, whatever library it comes from.

    Args:
        generator (Any): A numpy Generator or RandomState, a torch Generator or
            a Python random.Random.

    Returns:
        Any: A picklable copy of its state.
    """
    if isinstance(generator, np.random.Generator):
        return generator.bit_generator.state
    if isinstance(generator, (torch.Generator, np.random.RandomState)):
        return generator.get_state()
    return generator.getstate()


def set_generator_state(generator: Any, state: Any) -> None:
    """Put a random generator back in a state read by 'get_generator_state'.

    Args:
        generator (Any): The generator to restore, of the same kind as the one
            the state was read from.
        state (Any): The state.
    """
    if isinstance(generator, np.random.Generator):
        generator.bit_generator.state = state
    elif isinstance(generator, (torch.Generator, np.random.RandomState)):
        generator.set_state(state)
    else:
        generator.setstate(state)


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
    for holder in _rng_holders(getattr(info, "dataset", None)):
        holder.rng = np.random.default_rng(worker_seed)


class WorkerRandomState:
    """Collate function that keeps track of what persistent workers have drawn.

    A persistent worker keeps its copy of the dataset, and so its sampling
    generators, from one epoch to the next. That state lives in the worker
    process, out of reach of the checkpoint a resumed trial restarts from, so
    a resumed trial used to draw different negatives than the uninterrupted
    run from the first epoch it repeated onwards.

    The wrapper runs in the worker after every batch it builds. It writes the
    state of the worker's generators to a slot of shared memory, which the
    checkpoint reads at the end of an epoch, when every batch of the epoch has
    been delivered and the workers are idle.

    Restoring works the other way round. The state is handed to the workers
    when they start, but it must take effect only once they reach the first
    batch of the resumed epoch: the loader iterator Lightning creates before
    the first epoch has already given every worker its first 'prefetch_factor'
    batches, which a resumed trial discards. Each worker therefore counts the
    batches it builds and adopts the saved state after the last discarded one.

    Args:
        collate_fn (Optional[Callable[[Any], Any]]): The collate function to
            wrap. None stands for PyTorch's default collation.
        num_workers (int): The number of workers of the loader.

    Attributes:
        SLOT_BYTES (int): The shared memory given to the state of one worker.
    """

    SLOT_BYTES: int = 16384

    def __init__(self, collate_fn: Optional[Callable[[Any], Any]], num_workers: int):
        self.collate_fn = collate_fn or default_collate
        self.slots = torch.zeros((num_workers, self.SLOT_BYTES), dtype=torch.uint8)
        self.slots.share_memory_()
        self.pending: Optional[List[Optional[bytes]]] = None
        self.skip: List[int] = []
        self._built = 0

    def __call__(self, batch: Any) -> Any:
        """Collate a batch, then record the state of this worker's generators.

        Args:
            batch (Any): The samples of the batch.

        Returns:
            Any: The collated batch.
        """
        collated = self.collate_fn(batch)
        info = torch.utils.data.get_worker_info()
        if info is None:
            return collated

        holders = _rng_holders(info.dataset)
        self._built += 1
        if self.pending is not None and self._built == self.skip[info.id]:
            saved = self.pending[info.id]
            if saved is not None:
                states = pickle.loads(saved)  # nosec B301 - written by this loader
                for holder, state in zip(holders, states):
                    set_generator_state(holder.rng, state)
            self.pending = None

        record = pickle.dumps([get_generator_state(h.rng) for h in holders])
        slot = self.slots[info.id]
        if len(record) + 4 > self.SLOT_BYTES:
            slot[:4] = torch.tensor([255, 255, 255, 255], dtype=torch.uint8)
        else:
            header = len(record).to_bytes(4, "little")
            slot[: 4 + len(record)] = torch.frombuffer(
                bytearray(header + record), dtype=torch.uint8
            )
        return collated

    def snapshot(self) -> Optional[List[Optional[bytes]]]:
        """The last state each worker recorded.

        Returns:
            Optional[List[Optional[bytes]]]: One entry per worker, None for a
                worker that has built no batch yet. None when a state did not
                fit its slot and cannot be saved.
        """
        states: List[Optional[bytes]] = []
        for slot in self.slots:
            raw = bytes(slot.numpy())
            size = int.from_bytes(raw[:4], "little")
            if size == 0xFFFFFFFF:
                return None
            states.append(raw[4 : 4 + size] if size else None)
        return states

    def prime(self, states: List[Optional[bytes]], loader: DataLoader) -> None:
        """Hand saved states to the workers the loader is about to start.

        Args:
            states (List[Optional[bytes]]): The states 'snapshot' returned.
            loader (DataLoader): The loader, whose length and prefetch factor
                tell how many batches each worker builds before the epoch the
                trial resumes from.
        """
        workers = loader.num_workers
        first_requests = min(workers * (loader.prefetch_factor or 2), len(loader))
        self.skip = [len(range(w, first_requests, workers)) for w in range(workers)]
        self.pending = list(states)


def loader_random_state(loader: DataLoader) -> Dict[str, Any]:
    """Everything that decides the batches a training loader yields next.

    Args:
        loader (DataLoader): A loader built by 'seeded_dataloader'.

    Returns:
        Dict[str, Any]: The state of the shuffling generator, of the dataset's
            sampling generators in this process and, for persistent workers,
            of the copies the workers hold.
    """
    collate = loader.collate_fn
    return {
        "generator": (
            loader.generator.get_state() if loader.generator is not None else None
        ),
        "holders": [get_generator_state(h.rng) for h in _rng_holders(loader.dataset)],
        "workers": (
            collate.snapshot() if isinstance(collate, WorkerRandomState) else None
        ),
    }


def prime_loader_workers(loader: DataLoader, state: Dict[str, Any]) -> bool:
    """Give a loader's persistent workers the state saved by 'loader_random_state'.

    Must run before the workers start, that is before the loader is first
    iterated.

    Args:
        loader (DataLoader): The loader of the resumed trial.
        state (Dict[str, Any]): The saved state.

    Returns:
        bool: False when the workers' state was needed but could not be saved
            or restored, so that the resumed epochs will draw other samples.
    """
    collate = loader.collate_fn
    if not isinstance(collate, WorkerRandomState):
        return True
    if state.get("workers") is None or len(state["workers"]) != loader.num_workers:
        return False
    collate.prime(state["workers"], loader)
    return True


def restore_loader_random_state(loader: DataLoader, state: Dict[str, Any]) -> None:
    """Restore the shuffling and sampling state saved by 'loader_random_state'.

    Args:
        loader (DataLoader): The loader of the resumed trial.
        state (Dict[str, Any]): The saved state.
    """
    if loader.generator is not None and state.get("generator") is not None:
        loader.generator.set_state(state["generator"])
    for holder, saved in zip(_rng_holders(loader.dataset), state.get("holders", [])):
        set_generator_state(holder.rng, saved)


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

    # Persistent workers carry their sampling state across epochs, so it is
    # tracked for a trial that is paused and resumed.
    num_workers = kwargs.get("num_workers", 0)
    if num_workers > 0 and kwargs.get("persistent_workers", False):
        kwargs["collate_fn"] = WorkerRandomState(kwargs.get("collate_fn"), num_workers)

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

    A join does not promise to keep the rows in their input order (polars does
    not), so each row also carries its position in the input, ``ROW_INDEX``.
    Sorting on it after the indices keeps repeated (user, item) rows in file
    order, which is what the 'first' and 'last' duplicate policies mean.

    Args:
        data (DataFrame[Any]): The raw interactions.
        user_label (str): The name of the user column.
        item_label (str): The name of the item column.
        user_mapping (dict): Mapping of user ID -> user index.
        item_mapping (dict): Mapping of item ID -> item index.
        keep (Optional[list]): Columns to carry through in addition to the
            indices. Left out, every column of the input is kept.

    Returns:
        DataFrame[Any]: The frame with the index and row position columns attached.
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

    mapped = (
        data.with_row_index(ROW_INDEX)
        .join(umap_df, on=user_label, how="inner")
        .join(imap_df, on=item_label, how="inner")
    )

    if keep is None:
        return mapped
    return mapped.select(
        [nw.col(c) for c in (*keep, USER_INDEX, ITEM_INDEX, ROW_INDEX)]
    )
