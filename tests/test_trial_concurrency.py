"""How many trials a search runs at once follows the nodes that can host them."""

import pytest

from warprec.errors import ClusterError
from warprec.recommenders.trainer.trainer import trials_that_fit

NODES = [
    {"Alive": True, "Resources": {"CPU": 16.0, "GPU": 2.0}, "Labels": {"tier": "gpu"}},
    {"Alive": True, "Resources": {"CPU": 64.0}, "Labels": {"tier": "cpu"}},
    {"Alive": False, "Resources": {"CPU": 8.0, "GPU": 8.0}, "Labels": {"tier": "gpu"}},
]


def trial(per_worker, workers=1, label_selector=None):
    return {
        "num_workers": workers,
        "resources_per_worker": per_worker,
        "label_selector": label_selector,
    }


def test_only_the_nodes_that_can_host_a_worker_count():
    # The GPUs bound the GPU node, the dead node does not count, and the CPU
    # node has no GPU at all
    assert trials_that_fit(NODES, trial({"CPU": 2, "GPU": 1})) == 2
    assert (
        trials_that_fit(NODES, trial({"CPU": 4}, label_selector={"tier": "cpu"})) == 16
    )
    assert (
        trials_that_fit(NODES, trial({"CPU": 4}, label_selector={"tier": "!gpu"})) == 16
    )
    # Two workers of one GPU each make a single trial
    assert trials_that_fit(NODES, trial({"CPU": 1, "GPU": 1}, workers=2)) == 1


def test_a_trial_no_node_can_host_stops_the_search():
    with pytest.raises(ClusterError, match="No trial can run"):
        trials_that_fit(
            NODES, trial({"CPU": 1, "GPU": 1}, label_selector={"tier": "cpu"})
        )
