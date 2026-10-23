"""The sequential loaders read the process's own copy of the dataset.

A dataset fetched from Ray's object store holds read-only views of the stored
object, which Ray keeps on disk when the store cannot hold it; read at random,
one sample at a time, it made every sample a read from the disk.
"""

import copy

import torch

from warprec.data.dataset import Dataset


def test_a_loader_copies_the_arrays_it_would_read_from_the_object_store(
    dataset: Dataset,
):
    stored = copy.deepcopy(dataset.train_session)
    # What Ray hands a trial: the same values, read-only
    for array in (
        stored._flat_items,
        stored._flat_users,
        stored._user_offsets,
        stored._inter_sparse.indices,
        stored._inter_sparse.indptr,
    ):
        array.flags.writeable = False

    loader = stored.get_sequential_dataloader(max_seq_len=5, shuffle=False)
    read = loader.dataset
    assert read.flat_items.flags.writeable
    assert read.flat_users.flags.writeable
    assert read.sparse_matrix.indices.flags.writeable

    expected = dataset.train_session.get_sequential_dataloader(
        max_seq_len=5, shuffle=False
    )
    for ours, theirs in zip(next(iter(loader)), next(iter(expected))):
        assert torch.equal(ours, theirs)
