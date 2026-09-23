"""Behavioural tests for multimodal item features and the models that read them.

A feature matrix is anonymous: nothing in it says which item any row describes.
Everything here therefore turns on the join between the matrix and the row order
that names its items, and on what happens at the edges of that join — an item the
features miss, a row naming an item the catalogue dropped, two files that do not
agree on how many rows there are. The model tests ask what a multimodal model can
get silently wrong: reading a modality it was not given, and, for FREEDOM, whether
the two graphs it is built on are the shapes the paper says they are.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.data.entities import MultiModalFeatures
from warprec.data.reader import LocalReader
from warprec.utils.config import ModalityReading
from warprec.utils.registry import model_registry

from conftest import build_params

# The catalogue is written with identifiers that are not 0..n, because the
# mapping into index space is most of what these tests are about.
ITEM_MAPPING = {"b1": 0, "b2": 1, "b3": 2, "b4": 3}


@pytest.fixture(scope="module")
def features() -> MultiModalFeatures:
    """Features covering three of the four items, in a scrambled order.

    Returns:
        MultiModalFeatures: The features built against ITEM_MAPPING.
    """
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],  # b3
            [0.0, 2.0, 0.0],  # b1
            [0.0, 0.0, 3.0],  # gone, not in the catalogue
            [4.0, 4.0, 0.0],  # b2
        ],
        dtype="float32",
    )
    return MultiModalFeatures(
        {"visual": (matrix, ["b3", "b1", "gone", "b2"])}, ITEM_MAPPING
    )


def test_a_row_lands_on_the_item_its_order_file_names(features: MultiModalFeatures):
    """The join is positional against the row order, not against row number."""
    table = features.get("visual")

    assert torch.equal(table[0], torch.tensor([0.0, 2.0, 0.0]))  # b1
    assert torch.equal(table[1], torch.tensor([4.0, 4.0, 0.0]))  # b2
    assert torch.equal(table[2], torch.tensor([1.0, 0.0, 0.0]))  # b3


def test_an_item_the_features_miss_is_kept_and_reads_as_zero(
    features: MultiModalFeatures,
):
    """Dropping it would change the catalogue every model is compared on."""
    table = features.get("visual")

    # b4 is item 3 and no row describes it.
    assert torch.equal(table[3], torch.zeros(3))
    assert features.coverage()["visual"] == 3

    # It is still part of the catalogue, and the table still has a row for it.
    assert table.size(0) == len(ITEM_MAPPING) + 1


def test_the_table_carries_a_padding_row_past_the_catalogue(
    features: MultiModalFeatures,
):
    """Several families index one past the last item for a position holding nothing."""
    table = features.get("visual")

    assert torch.equal(table[len(ITEM_MAPPING)], torch.zeros(3))


def test_a_row_naming_an_item_outside_the_catalogue_is_ignored(
    features: MultiModalFeatures,
):
    """A feature dump is written for a whole dataset, not for one split."""
    table = features.get("visual")

    # The vector written for "gone" must not have landed anywhere.
    assert not torch.any(torch.all(table == torch.tensor([0.0, 0.0, 3.0]), dim=1))


def test_features_and_their_row_order_must_agree():
    """Two files of different lengths cannot describe the same modality."""
    with pytest.raises(ValueError, match="do not describe"):
        MultiModalFeatures(
            {"visual": (np.zeros((3, 2), dtype="float32"), ["b1", "b2"])},
            ITEM_MAPPING,
        )


def test_normalisation_leaves_an_uncovered_item_at_zero():
    """An item the features miss must not become an arbitrary unit vector."""
    matrix = np.array([[3.0, 4.0], [6.0, 8.0]], dtype="float32")

    features = MultiModalFeatures(
        {"visual": (matrix, ["b1", "b2"])}, ITEM_MAPPING, normalize={"visual": "l2"}
    )
    table = features.get("visual")

    assert pytest.approx(float(table[0].norm()), abs=1e-6) == 1.0
    assert pytest.approx(float(table[1].norm()), abs=1e-6) == 1.0
    # b3, b4 and the padding row are all uncovered.
    assert float(table[2:].norm()) == 0.0


def test_the_modalities_keep_their_own_widths():
    """Nothing forces two modalities into a common width."""
    features = MultiModalFeatures(
        {
            "visual": (np.zeros((2, 12), dtype="float32"), ["b1", "b2"]),
            "textual": (np.zeros((2, 5), dtype="float32"), ["b1", "b2"]),
        },
        ITEM_MAPPING,
    )

    assert features.dims() == {"visual": 12, "textual": 5}
    assert features.names() == ["visual", "textual"]
    assert "visual" in features and "audio" not in features


def test_reading_an_array_gives_one_row_per_item(tmp_path):
    """A .npy is the format the published features are shipped in."""
    path = tmp_path / "feat.npy"
    np.save(path, np.arange(6, dtype="float64").reshape(3, 2))

    array = LocalReader().read_array(local_path=str(path))

    assert array.shape == (3, 2)
    assert array.dtype == np.float32


def test_a_single_feature_per_item_is_still_a_matrix(tmp_path):
    """A one-dimensional file is a catalogue of scalars, not one long vector."""
    path = tmp_path / "feat.npy"
    np.save(path, np.arange(4, dtype="float32"))

    array = LocalReader().read_array(local_path=str(path))

    assert array.shape == (4, 1)


def test_an_array_of_features_must_say_which_item_each_row_is():
    """A matrix silently offset against the catalogue would train without complaint."""
    with pytest.raises(ValueError, match="item_path"):
        ModalityReading(local_path="feat.npy")

    # A tabular file carries the identifiers in the file itself.
    assert (
        ModalityReading(local_path="feat.tsv", file_format="tabular").item_path is None
    )


def build(model_name: str, dataset: Dataset, **overrides: Any) -> Any:
    """Construct one multimodal model against the shared fixture.

    Args:
        model_name (str): The registered name.
        dataset (Dataset): The dataset under test.
        **overrides (Any): Hyperparameters to change from the smoke defaults.

    Returns:
        Any: The constructed model.
    """
    params: Dict[str, Any] = {**build_params(model_name), **overrides}
    return model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        multimodal=dataset.multimodal,
    )


@pytest.mark.parametrize("model_name", ["VBPR", "FREEDOM"])
def test_a_multimodal_model_refuses_a_dataset_without_features(
    model_name: str, dataset: Dataset
):
    """Scoring from features that are not there would quietly become collaborative."""
    with pytest.raises(ValueError, match="multimodal item features"):
        model_registry.get(
            model_name,
            params=build_params(model_name),
            info=dataset.info(),
            interactions=dataset.train_set,
            sessions=dataset.train_session,
            transactions=dataset.train_transactions,
            multimodal=None,
        )


@pytest.mark.parametrize("model_name", ["VBPR", "FREEDOM"])
def test_a_model_refuses_a_modality_the_dataset_does_not_carry(
    model_name: str, dataset: Dataset
):
    """A misspelled modality must not silently become an empty feature block."""
    with pytest.raises(ValueError, match="audio"):
        build(model_name, dataset, modalities=["audio"])


def test_leaving_the_modalities_unset_reads_all_of_them(dataset: Dataset):
    """A configuration that adds a modality should not leave it unused."""
    model = build("VBPR", dataset)

    assert model.modality_names == dataset.multimodal.names()
    assert model.joint_features().size(1) == sum(dataset.multimodal.dims().values())


def test_naming_one_modality_reads_only_that_one(dataset: Dataset):
    """The subset is what makes VBPR reduce to the single-modality paper."""
    model = build("VBPR", dataset, modalities=["visual"])

    assert model.modality_names == ["visual"]
    assert model.joint_features().size(1) == dataset.multimodal.dims()["visual"]


@pytest.mark.parametrize("model_name", ["VBPR", "FREEDOM"])
def test_what_a_model_is_given_is_read_only(model_name: str, dataset: Dataset):
    """The features as the file wrote them are data, not parameters.

    FREEDOM refines a working copy of them, but the copy the item-item graph was
    built from must stay where it is, or the graph stops being the frozen thing
    the paper argues for.
    """
    model = build(model_name, dataset)
    parameters = dict(model.named_parameters())

    for name in model.modality_names:
        assert f"modality_{name}" not in parameters
        assert not model.modality(name).requires_grad
        # They still travel with the checkpoint, so serving does not need the
        # feature files to still be where they were at training time.
        assert f"modality_{name}" in model.state_dict()


def test_vbpr_learns_only_the_projection_out_of_the_features(dataset: Dataset):
    """The paper treats the feature vector as a constant."""
    model = build("VBPR", dataset)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert "feature_projection.weight" in trainable
    assert not any(name.startswith("modality_") for name in trainable)


def test_freedom_refines_a_copy_of_the_features(dataset: Dataset):
    """The reference implementation updates the vectors as well as the projection."""
    model = build("FREEDOM", dataset)
    refined = {name for name, p in model.named_parameters() if p.requires_grad}

    assert any(name.startswith("refined.") for name in refined)

    # It starts as a copy of what the file held, and the original does not move
    # with it.
    for index, name in enumerate(model.modality_names):
        assert torch.equal(model.refined[index].weight.detach(), model.modality(name))

    graph_before = model.item_item.values().clone()
    with torch.no_grad():
        model.refined[0].weight.mul_(4.0)

    assert not torch.equal(
        model.refined[0].weight.detach(), model.modality(model.modality_names[0])
    )
    assert torch.equal(graph_before, model.item_item.values())


def test_the_frozen_item_graph_keeps_one_neighbourhood_per_item(dataset: Dataset):
    """FREEDOM's item-item graph is a kNN graph, built once and never learned."""
    knn_k = 3
    model = build("FREEDOM", dataset, knn_k=knn_k)
    n_items = dataset.info()["n_items"]

    graph = model.item_item
    assert graph.is_sparse
    assert graph.shape == (n_items + 1, n_items + 1)

    # Two modalities each vote for knn_k neighbours per item, and where they
    # agree the votes coalesce, so this is an upper bound rather than equality.
    per_item = torch.bincount(graph.indices()[0], minlength=n_items + 1)
    assert int(per_item[:n_items].max()) <= knn_k * len(dataset.multimodal)
    assert int(per_item[:n_items].min()) >= knn_k

    # The padding row describes nothing, so it has no neighbours.
    assert int(per_item[n_items]) == 0


def test_the_modality_weights_change_the_frozen_graph(dataset: Dataset):
    """Each modality's graph is normalised before it is weighted.

    Normalising the weighted sum instead would divide the weights straight back
    out again, leaving the parameter with almost no effect. The two settings
    below must therefore disagree.
    """
    even = build("FREEDOM", dataset, modality_weights=[0.5, 0.5])
    lopsided = build("FREEDOM", dataset, modality_weights=[0.05, 0.95])

    assert torch.equal(even.item_item.indices(), lopsided.item_item.indices())
    assert not torch.allclose(even.item_item.values(), lopsided.item_item.values())


def test_a_weight_is_needed_for_every_modality(dataset: Dataset):
    """Silently zipping the shorter list would drop a modality without a word."""
    with pytest.raises(ValueError, match="one weight per modality"):
        build("FREEDOM", dataset, modality_weights=[0.5])


def test_the_frozen_item_graph_does_not_change_while_training(dataset: Dataset):
    """Freezing it is the paper's whole argument; relearning it would be a different model."""
    model = build("FREEDOM", dataset)
    before = model.item_item.values().clone()

    model.on_train_epoch_start()
    with torch.no_grad():
        model.item_embedding.weight.mul_(3.0)

    assert torch.equal(before, model.item_item.values())


def test_dropping_edges_thins_the_interaction_graph(dataset: Dataset):
    """The user-item graph is denoised afresh each epoch, and must actually shrink."""
    model = build("FREEDOM", dataset, dropout=0.5)
    full = model.adj.nnz()

    model.train()
    model.on_train_epoch_start()

    assert model.masked_adj.nnz() < full


def test_keeping_every_edge_is_the_whole_graph(dataset: Dataset):
    """A run configured not to denoise must see all of its interactions."""
    model = build("FREEDOM", dataset, dropout=0.0)

    model.on_train_epoch_start()

    assert model.masked_adj.nnz() == model.adj.nnz()


def test_the_dataset_publishes_the_widths_the_models_size_themselves_from(
    dataset: Dataset, multimodal_frames: Dict[str, Dict[str, Any]]
):
    """A model reads the width of each modality out of the dataset info."""
    info = dataset.info()

    assert dataset.multimodal is not None
    assert info["modality_dims"] == dataset.multimodal.dims()
    assert info["modality_dims"] == {
        name: payload["features"].shape[1]
        for name, payload in multimodal_frames.items()
    }
