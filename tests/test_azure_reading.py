"""Reading every kind of input from Azure Blob Storage, with the client mocked.

The configuration names its blobs under 'azure_blob_name' and 'azure_blob_prefix',
and these tests check that each of them reaches the reader: the dataset, the
pre-split files, the side information, the knowledge graph, the multimodal
features and the cluster files. No network is touched.
"""

import io
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

pytest.importorskip("azure.storage.blob")

# pylint: disable=wrong-import-position
from azure.core.exceptions import ResourceNotFoundError  # noqa: E402

from warprec.common.initialize import initialize_datasets  # noqa: E402
from warprec.data.reader import ReaderFactory  # noqa: E402
from warprec.data.reader import azureblob_reader  # noqa: E402
from warprec.utils.callback import WarpRecCallback  # noqa: E402
from warprec.utils.config.config import TrainConfiguration  # noqa: E402
from warprec.utils.config.reader_configuration import ReaderConfig  # noqa: E402

N_USERS, N_ITEMS = 10, 12


class FakeDownload:
    """What download_blob returns: the blob's bytes.

    Args:
        content (bytes): The blob's content.
    """

    def __init__(self, content: bytes):
        self.content = content

    def readall(self) -> bytes:
        """Return the blob's content.

        Returns:
            bytes: The content.
        """
        return self.content


class FakeBlob:
    """A blob client over an in-memory store.

    Args:
        store (Dict[str, bytes]): The blobs of the container.
        name (str): The name of this blob.
    """

    def __init__(self, store: Dict[str, bytes], name: str):
        self.store = store
        self.name = name

    def download_blob(self) -> FakeDownload:
        """Download the blob, as the Azure client does.

        Returns:
            FakeDownload: The download.

        Raises:
            ResourceNotFoundError: If the container has no such blob.
        """
        if self.name not in self.store:
            raise ResourceNotFoundError(f"No blob {self.name}")
        return FakeDownload(self.store[self.name])


class FakeContainer:
    """A container client over an in-memory store.

    Args:
        store (Dict[str, bytes]): The blobs of the container.
    """

    def __init__(self, store: Dict[str, bytes]):
        self.store = store

    def get_blob_client(self, name: str) -> FakeBlob:
        """Return the client of one blob.

        Args:
            name (str): The name of the blob.

        Returns:
            FakeBlob: Its client.
        """
        return FakeBlob(self.store, name)


def tsv(frame: pd.DataFrame, header: bool = True) -> bytes:
    """Serialise a frame as a tab-separated blob.

    Args:
        frame (pd.DataFrame): The frame.
        header (bool): Whether the blob carries a header row.

    Returns:
        bytes: The blob content.
    """
    return frame.to_csv(sep="\t", index=False, header=header).encode("utf-8")


def npy(array: np.ndarray) -> bytes:
    """Serialise an array as a .npy blob.

    Args:
        array (np.ndarray): The array.

    Returns:
        bytes: The blob content.
    """
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


@pytest.fixture
def blobs(monkeypatch: pytest.MonkeyPatch) -> Dict[str, bytes]:
    """A container holding every kind of input, served by a mocked Azure client.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the Azure client.

    Returns:
        Dict[str, bytes]: The blobs, by name.
    """
    rng = np.random.default_rng(0)
    rows = [
        (user, int(item), 1.0, user * 100 + rank)
        for user in range(N_USERS)
        for rank, item in enumerate(rng.choice(N_ITEMS, 6, replace=False))
    ]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    test = ratings.groupby("user_id").tail(1)
    items = pd.DataFrame({"item_id": range(N_ITEMS)})

    store = {
        "data/ratings.tsv": tsv(ratings),
        "split/train.tsv": tsv(ratings.drop(test.index)),
        "split/test.tsv": tsv(test),
        "data/side.tsv": tsv(items.assign(genre=[i % 3 for i in range(N_ITEMS)])),
        "data/kg.tsv": tsv(
            pd.DataFrame({"head": range(N_ITEMS), "relation": 0, "tail": N_ITEMS + 1}),
            header=False,
        ),
        "data/links.tsv": tsv(
            pd.DataFrame({"item_id": range(N_ITEMS), "entity_id": range(N_ITEMS)}),
            header=False,
        ),
        "data/visual.npy": npy(rng.random((N_ITEMS, 4)).astype(np.float32)),
        "data/visual_items.tsv": tsv(items, header=False),
        "data/textual.tsv": tsv(
            items.assign(a=rng.random(N_ITEMS), b=rng.random(N_ITEMS)), header=False
        ),
        "data/user_clusters.tsv": tsv(
            pd.DataFrame({"user_id": range(N_USERS), "cluster": 1})
        ),
        "data/item_clusters.tsv": tsv(
            pd.DataFrame({"item_id": range(N_ITEMS), "cluster": 2})
        ),
    }

    class FakeService:
        """A blob service client serving the store."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def get_container_client(self, name: str) -> FakeContainer:
            """Return the container.

            Args:
                name (str): The container name.

            Returns:
                FakeContainer: The container.
            """
            return FakeContainer(store)

    monkeypatch.setattr(azureblob_reader, "BlobServiceClient", FakeService)
    monkeypatch.setattr(azureblob_reader, "DefaultAzureCredential", lambda: None)
    return store


def configuration(tmp_path: Path, reader: Dict[str, Any]) -> TrainConfiguration:
    """Validate a training configuration reading from Azure.

    Args:
        tmp_path (Path): Where the outputs would be written.
        reader (Dict[str, Any]): The reader section, beyond the common keys.

    Returns:
        TrainConfiguration: The validated configuration.
    """
    return TrainConfiguration(
        reader={
            "data_type": "transaction",
            "reading_method": "azure_blob",
            "rating_type": "implicit",
            **reader,
        },
        general={"azure": {"storage_account_name": "acc", "container_name": "c"}},
        writer={
            "dataset_name": "azure",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path / "out"),
        },
        models={"Pop": {}},
        evaluation={"top_k": [5], "metrics": ["nDCG"]},
        splitter={"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
    )


def test_every_input_is_read_from_its_blob(tmp_path: Path, blobs: Dict[str, bytes]):
    """The dataset and every optional input are read from the blobs they name."""
    config = configuration(
        tmp_path,
        {
            "loading_strategy": "dataset",
            "azure_blob_name": "data/ratings.tsv",
            "side": {"azure_blob_name": "data/side.tsv"},
            "knowledge": {
                "azure_blob_name": "data/kg.tsv",
                "link_azure_blob_name": "data/links.tsv",
            },
            "multimodal": {
                "visual": {
                    "azure_blob_name": "data/visual.npy",
                    "item_azure_blob_name": "data/visual_items.tsv",
                },
                "textual": {
                    "azure_blob_name": "data/textual.tsv",
                    "file_format": "tabular",
                },
            },
            "clustering": {
                "user_azure_blob_name": "data/user_clusters.tsv",
                "item_azure_blob_name": "data/item_clusters.tsv",
            },
        },
    )

    main, _, _ = initialize_datasets(
        ReaderFactory.get_reader(config), WarpRecCallback(), config
    )

    interactions = main.train_set.get_sparse().nnz + main.eval_set.get_sparse().nnz
    assert interactions == N_USERS * 6
    assert set(main.user_cluster.values()) == {1}
    assert set(main.item_cluster.values()) == {2}
    assert main.get_feature_matrix() is not None
    assert main.knowledge is not None
    assert main.multimodal.dims() == {"visual": 4, "textual": 2}


def test_pre_split_files_are_read_from_their_prefix(
    tmp_path: Path, blobs: Dict[str, bytes]
):
    """The split files are read from under 'split.azure_blob_prefix'."""
    config = configuration(
        tmp_path,
        {"loading_strategy": "split", "split": {"azure_blob_prefix": "split"}},
    )

    main, validation, _ = initialize_datasets(
        ReaderFactory.get_reader(config), WarpRecCallback(), config
    )

    assert validation is None
    assert main.eval_set.get_sparse().nnz == N_USERS


@pytest.mark.parametrize(
    "reader, message",
    [
        ({"loading_strategy": "dataset"}, "azure_blob_name"),
        ({"loading_strategy": "split"}, "azure_blob_prefix"),
    ],
)
def test_a_missing_blob_location_is_refused_at_load(
    reader: Dict[str, Any], message: str
):
    """An Azure reader with no blob to read fails when it is loaded, naming the key."""
    with pytest.raises(ValidationError, match=message):
        ReaderConfig(
            data_type="transaction",
            reading_method="azure_blob",
            rating_type="implicit",
            **reader,
        )
