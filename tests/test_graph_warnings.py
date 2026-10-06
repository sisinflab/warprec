"""Graph models must not flood every log with PyTorch Geometric's internal notices.

Importing PyG runs torch.jit.script in its own pooling layers, and multiplying
through its EdgeIndex builds a sparse CSR tensor; PyTorch warns about both,
once per process. Each Ray worker and Serve replica is a process of its own, so
the same notices repeated across every log. The probe runs in a fresh process
with every warning enabled, so that nothing earlier can hide them.
"""

import subprocess
import sys

import pytest

pytest.importorskip("torch_geometric")

PROBE = """
import torch
from warprec.recommenders.collaborative_filtering_recommender.graph_based.graph_utils import (
    SparseAdjacency,
)

adjacency = SparseAdjacency(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), size=(3, 3))
# Training multiplies embeddings that need gradients: PyG builds another sparse
# tensor in the backward pass, outside the forward call.
embeddings = torch.ones(3, 2, requires_grad=True)
adjacency.matmul(embeddings).sum().backward()
print(adjacency.matmul(torch.ones(3, 2)).tolist())
"""


@pytest.fixture(scope="module")
def probe() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_the_product_is_still_computed(probe: subprocess.CompletedProcess):
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert (
        probe.stdout.strip().splitlines()[-1] == "[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]"
    )


@pytest.mark.parametrize(
    "notice",
    [
        "`torch.jit.script` is deprecated",
        "Sparse CSR tensor support is in beta state",
        "Sparse invariant checks are implicitly disabled",
    ],
)
def test_pyg_internals_raise_no_warning(
    probe: subprocess.CompletedProcess, notice: str
):
    assert notice not in probe.stderr
