import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# The guides download their data from its publisher rather than shipping it,
# because most recommendation datasets may not be redistributed.
MOVIELENS_100K = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
KGCN_MUSIC = "https://raw.githubusercontent.com/hwwang55/KGCN/master/data/music/"
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]  # fmt: skip


def _download(url: str) -> bytes:
    """Fetch a URL into memory.

    Args:
        url (str): One of the fixed https addresses above.

    Returns:
        bytes: The response body.
    """
    with urllib.request.urlopen(url) as response:  # nosec B310 (fixed https URL)
        return response.read()


def movielens_100k(data_dir: str | Path = "../data") -> Path:
    """Download MovieLens-100K once and write it in the layout the guides read.

    This is what the first guide does step by step: the ratings become a
    tab-separated file with a header, the item and user files become tables with
    named columns, and the genre flags become one '|'-separated column.

    Args:
        data_dir (str | Path): The shared download cache of the guides.

    Returns:
        Path: The directory holding ratings.tsv, items.tsv, users.tsv and the
            original files.
    """
    folder = Path(data_dir) / "ml-100k"
    if (folder / "ratings.tsv").exists():
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(_download(MOVIELENS_100K))).extractall(folder.parent)

    ratings = pd.read_csv(
        folder / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    ratings.to_csv(folder / "ratings.tsv", sep="\t", index=False)

    items = pd.read_csv(folder / "u.item", sep="|", header=None, encoding="latin-1")
    flags = items.iloc[:, 5:].to_numpy().astype(bool)
    pd.DataFrame(
        {
            "item_id": items[0],
            "title": items[1],
            "release_date": items[2],
            "genres": [
                "|".join(g for g, on in zip(GENRES, row) if on) for row in flags
            ],
        }
    ).to_csv(folder / "items.tsv", sep="\t", index=False)

    users = pd.read_csv(
        folder / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    users.to_csv(folder / "users.tsv", sep="\t", index=False)
    return folder


def kgcn_music(data_dir: str | Path = "../data") -> Path:
    """Download the LastFM data and knowledge graph released with KGCN.

    Args:
        data_dir (str | Path): The shared download cache of the guides.

    Returns:
        Path: The directory holding user_artists.dat, kg.txt and
            item_index2entity_id.txt.
    """
    folder = Path(data_dir) / "kgcn-music"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("user_artists.dat", "kg.txt", "item_index2entity_id.txt"):
        if not (folder / name).exists():
            (folder / name).write_bytes(_download(KGCN_MUSIC + name))
    return folder
