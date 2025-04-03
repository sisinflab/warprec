from abc import ABC, abstractmethod

from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from elliotwo.utils.enums import Similarities
from elliotwo.utils.registry import similarities_registry


class Similarity(ABC):
    """Abstract definition of a similarity measure."""

    @abstractmethod
    def compute(self, X: csr_matrix) -> ndarray:
        """Compute the similarity"""


@similarities_registry.register(Similarities.COSINE)
class Cosine(Similarity):
    def compute(self, X: csr_matrix) -> ndarray:
        return cosine_similarity(X)


@similarities_registry.register(Similarities.DOT)
class Dot(Similarity):
    def compute(self, X: csr_matrix) -> ndarray:
        return (X @ X.T).toarray()


@similarities_registry.register(Similarities.EUCLIDEAN)
class Euclidean(Similarity):
    def compute(self, X: csr_matrix) -> ndarray:
        return 1 / (1 + euclidean_distances(X))


@similarities_registry.register(Similarities.MANHATTAN)
class Manhattan(Similarity):
    def compute(self, X: csr_matrix) -> ndarray:
        return 1 / (1 + manhattan_distances(X))
