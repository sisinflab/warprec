"""A positive can be contrasted with several negatives, and one stays the default."""

import numpy as np
import torch

from warprec.data.dataset import Dataset
from warprec.data.entities.train_structures.interaction_structures import (
    NegativeSampler,
)

from conftest import make_model


def test_several_negatives_are_items_the_user_has_not_seen(dataset: Dataset):
    train = dataset.train_set.get_sparse()
    sampler = NegativeSampler(train, train.shape[1], seed=3)
    for user in range(train.shape[0]):
        drawn = sampler.sample_many(user, 32)
        assert len(drawn) == 32
        assert not np.isin(drawn, train[user].indices).any()


def test_the_loader_yields_a_row_of_negatives_per_positive(dataset: Dataset):
    one = dataset.train_set.get_contrastive_dataloader(batch_size=8, shuffle=False)
    many = dataset.train_set.get_contrastive_dataloader(
        batch_size=8, shuffle=False, neg_samples=4
    )
    users, positives, negatives = next(iter(one))
    assert negatives.shape == (8,)
    many_users, many_positives, many_negatives = next(iter(many))
    assert many_negatives.shape == (8, 4)
    assert torch.equal(users, many_users) and torch.equal(positives, many_positives)


def test_bpr_trains_on_several_negatives(dataset: Dataset):
    model = make_model("BPR", dataset)
    model.neg_samples = 4
    batch = next(
        iter(
            model.get_dataloader(
                interactions=dataset.train_set, sessions=dataset.train_session
            )
        )
    )
    assert batch[2].shape[1] == 4
    assert torch.isfinite(model.training_step(batch, 0))
