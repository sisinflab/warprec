from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from warprec.utils.logger import logger


class MultiModalFeatures:
    """What is known about the items besides who interacted with them.

    A modality is a matrix of precomputed vectors, one row per item: a picture
    put through a convolutional network, a description put through a sentence
    encoder. The matrix carries no identifiers, so it arrives with the list of
    items its rows describe, and this class is what turns that pairing into
    something indexable by item index like everything else in the data layer.

    An item the features do not cover keeps its interactions and is given a row
    of zeros. It is not dropped, because a feature dump that misses a few items
    would otherwise change the catalogue every model in the run is compared on,
    for a reason that has nothing to do with any of them.

    Args:
        features (Dict[str, Tuple[np.ndarray, Sequence[Any]]]): Per modality, the
            matrix of vectors and the item each of its rows describes.
        item_mapping (dict): Mapping of item ID -> item index, from the dataset.
        normalize (Optional[Dict[str, str]]): Per modality, what to do to each
            row before the models see it. Only 'l2' does anything.

    Attributes:
        n_items (int): How many items the catalogue holds.
    """

    n_items: int

    def __init__(
        self,
        features: Dict[str, Tuple[np.ndarray, Sequence[Any]]],
        item_mapping: dict,
        normalize: Optional[Dict[str, str]] = None,
    ):
        self.n_items = len(item_mapping)
        self._features: Dict[str, Tensor] = {}
        self._covered: Dict[str, int] = {}
        normalize = normalize or {}

        for name, (matrix, row_items) in features.items():
            table, covered = self._align(matrix, row_items, item_mapping)

            if normalize.get(name) == "l2":
                # The padding row stays zero: normalising it would turn an item
                # the features miss into an arbitrary unit vector.
                table = torch.nn.functional.normalize(table, p=2, dim=1)

            self._features[name] = table
            self._covered[name] = covered

        if self._features:
            described = "      ".join(
                f"{name}: {table.size(1)} dims"
                for name, table in self._features.items()
            )
            coverage = "      ".join(
                f"{name}: {covered}/{self.n_items}"
                for name, covered in self._covered.items()
            )
            logger.stat_msg(
                f"{described}\nItems with features   {coverage}", "Multimodal features"
            )

    @staticmethod
    def _align(
        matrix: np.ndarray,
        row_items: Sequence[Any],
        item_mapping: dict,
    ) -> Tuple[Tensor, int]:
        """Put the rows of one modality in catalogue order.

        Args:
            matrix (np.ndarray): The feature vectors, one row per named item.
            row_items (Sequence[Any]): The item each row describes.
            item_mapping (dict): Mapping of item ID -> item index.

        Returns:
            Tuple[Tensor, int]: The {(item + padding) x feature} table, and how
                many items of the catalogue it actually covers.

        Raises:
            ValueError: If the matrix and its row order disagree on how many
                rows there are.
        """
        if len(row_items) != matrix.shape[0]:
            raise ValueError(
                f"The features hold {matrix.shape[0]} rows but their row order "
                f"names {len(row_items)} items. The two files do not describe "
                "the same modality."
            )

        # One row past the catalogue is the padding item, which several model
        # families index for a position that holds nothing.
        table = torch.zeros((len(item_mapping) + 1, matrix.shape[1]))

        positions, rows = [], []
        for row, item in enumerate(row_items):
            position = item_mapping.get(item)
            if position is not None:
                positions.append(position)
                rows.append(row)

        if positions:
            table[torch.as_tensor(positions, dtype=torch.long)] = torch.as_tensor(
                matrix[rows], dtype=torch.float
            )

        return table, len(set(positions))

    def names(self) -> List[str]:
        """The modalities that were configured.

        Returns:
            List[str]: The modality names, in the order they were configured.
        """
        return list(self._features)

    def dims(self) -> Dict[str, int]:
        """How wide each modality is.

        Returns:
            Dict[str, int]: The width of each modality's vectors.
        """
        return {name: table.size(1) for name, table in self._features.items()}

    def coverage(self) -> Dict[str, int]:
        """How many items each modality actually describes.

        Returns:
            Dict[str, int]: The number of covered items per modality.
        """
        return dict(self._covered)

    def get(self, name: str) -> Tensor:
        """One modality's features, in catalogue order.

        Args:
            name (str): The modality to read.

        Returns:
            Tensor: The {(item + padding) x feature} table.

        Raises:
            KeyError: If that modality was not configured.
        """
        if name not in self._features:
            raise KeyError(
                f"The modality '{name}' was not configured. The configured ones "
                f"are {self.names()}."
            )
        return self._features[name]

    def select(self, names: Optional[Sequence[str]] = None) -> List[Tensor]:
        """The features of several modalities at once.

        Args:
            names (Optional[Sequence[str]]): The modalities to read. Defaults to
                every configured one, in configuration order.

        Returns:
            List[Tensor]: One table per requested modality.
        """
        return [
            self.get(name) for name in (names if names is not None else self.names())
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)
