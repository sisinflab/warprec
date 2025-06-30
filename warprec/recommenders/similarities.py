# pylint: disable = too-few-public-methods
from abc import ABC, abstractmethod
from typing import Optional

from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    haversine_distances,
)
from warprec.utils.enums import Similarities
from warprec.utils.registry import similarities_registry


class Similarity(ABC):
    """Abstract definition of a similarity measure."""

    @abstractmethod
    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        """Compute the similarity.

        Args:
            X (csr_matrix): The interaction matrix.
            Y (Optional[csr_matrix]): Optional secondary matrix.

        Returns:
            ndarray: The similarity matrix.
        """


@similarities_registry.register(Similarities.COSINE)
class Cosine(Similarity):
    """Cosine similarity wrapper."""

    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        if Y is not None:
            return cosine_similarity(X, Y)
        return cosine_similarity(X)


@similarities_registry.register(Similarities.DOT)
class Dot(Similarity):
    """Dot similarity wrapper."""

    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        if Y is not None:
            return (X @ Y.T).toarray()
        return (X @ X.T).toarray()


@similarities_registry.register(Similarities.EUCLIDEAN)
class Euclidean(Similarity):
    """Euclidean similarity wrapper."""

    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        if Y is not None:
            return 1 / (1 + euclidean_distances(X, Y))
        return 1 / (1 + euclidean_distances(X))


@similarities_registry.register(Similarities.MANHATTAN)
class Manhattan(Similarity):
    """Manhattan similarity wrapper."""

    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        if Y is not None:
            return 1 / (1 + manhattan_distances(X, Y))
        return 1 / (1 + manhattan_distances(X))


@similarities_registry.register(Similarities.HAVERSINE)
class Haversine(Similarity):
    """Haversine similarity wrapper."""

    def compute(self, X: csr_matrix, Y: Optional[csr_matrix] = None) -> ndarray:
        if Y is not None:
            return 1 / (1 + haversine_distances(X, Y))
        return 1 / (1 + haversine_distances(X))
