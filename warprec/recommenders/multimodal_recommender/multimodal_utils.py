from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from warprec.data.entities import MultiModalFeatures


class MultiModalRecommenderUtils(nn.Module):
    """Common ground for the models that score from precomputed item features.

    Every model here needs the same thing: a frozen table of vectors per item,
    already aligned to the catalogue, for whichever modalities it was asked to
    read. The features are registered as buffers rather than parameters because
    nothing here fine-tunes an encoder; what is learned is the projection out of
    them, which is what the literature these models come from does.

    A model may be given a subset of the configured modalities through the
    'modalities' parameter. Left unset it reads all of them, so a configuration
    that adds a modality does not silently leave it unused.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        modality_names (List[str]): The modalities this model actually reads.
        modality_dims (List[int]): The width of each of those modalities.
        n_items (int): The number of items, from the recommender base class.
        n_users (int): The number of users, from the recommender base class.

    Raises:
        ValueError: If no features were provided, or a named modality is absent.
    """

    modality_names: List[str]
    modality_dims: List[int]

    # Set by the recommender base class; declared here so that the helpers below
    # read them as the integers they are rather than as the union a module
    # attribute is typed with.
    n_items: int
    n_users: int

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        multimodal: Optional[MultiModalFeatures] = None,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, **kwargs)  # type: ignore[call-arg]

        if multimodal is None or not len(multimodal):
            raise ValueError(
                f"{type(self).__name__} scores from multimodal item features, but "
                "the dataset carries none. Configure 'reader.multimodal'."
            )

        wanted = getattr(self, "modalities", None) or multimodal.names()
        if isinstance(wanted, str):
            wanted = [wanted]

        missing = [name for name in wanted if name not in multimodal]
        if missing:
            raise ValueError(
                f"{type(self).__name__} was asked for the modalities {missing}, "
                f"which the dataset does not carry. It carries "
                f"{multimodal.names()}."
            )

        self.modality_names = list(wanted)
        self.modality_dims = [multimodal.dims()[name] for name in self.modality_names]

        # One buffer per modality, so a checkpoint carries the features it was
        # trained against and a reload does not depend on the files still being
        # where they were.
        for name, table in zip(
            self.modality_names, multimodal.select(self.modality_names)
        ):
            self.register_buffer(f"modality_{name}", table, persistent=True)

    @torch.no_grad()
    def feature_neighbour_graph(
        self, how_many: int, weights: Optional[Sequence[float]] = None
    ) -> Tensor:
        """Build one item-item graph out of the features.

        Each modality votes for the neighbours of every item by cosine
        similarity. Each modality's graph is normalised on its own and only then
        weighted and summed, which is what makes the weights mean anything: a
        graph normalised after the sum would divide the weights straight back
        out again. The similarity is taken in blocks, because the full matrix is
        quadratic in the catalogue and is never needed at once.

        Args:
            how_many (int): How many neighbours each item keeps.
            weights (Optional[Sequence[float]]): How much each modality counts.
                Defaults to equal weight.

        Returns:
            Tensor: The sparse, normalised item-item adjacency.

        Raises:
            ValueError: If there is not one weight per modality.
        """
        if weights is not None and len(weights) != len(self.modality_names):
            raise ValueError(
                f"{type(self).__name__} was given {len(weights)} modality weights "
                f"for {len(self.modality_names)} modalities "
                f"({self.modality_names}). There must be one weight per modality, "
                "in the same order."
            )

        share = weights or [1.0 / len(self.modality_names)] * len(self.modality_names)
        side = self.n_items + 1
        combined: Optional[Tensor] = None

        for table, weight in zip(self.modality_tables(), share):
            # The padding row holds nothing, so it takes part in no similarity.
            features = F.normalize(table[: self.n_items], p=2, dim=1)
            neighbours = self._nearest_neighbours(features, how_many)

            rows = torch.arange(self.n_items).unsqueeze(1).expand(-1, how_many)
            indices = torch.stack([rows.flatten(), neighbours.flatten()])

            one_graph = self._normalise_neighbour_graph(indices, side) * weight
            combined = one_graph if combined is None else combined + one_graph

        return combined.coalesce()

    @staticmethod
    def _nearest_neighbours(features: Tensor, how_many: int) -> Tensor:
        """Which items each item is closest to.

        Args:
            features (Tensor): The row-normalised features of the catalogue.
            how_many (int): How many neighbours to keep.

        Returns:
            Tensor: The {item x how_many} neighbour indices.
        """
        # A block of 2048 rows against the whole catalogue is a few hundred
        # megabytes at most, whatever the catalogue size.
        block = 2048
        found = []
        for start in range(0, features.size(0), block):
            similarity = features[start : start + block] @ features.t()
            found.append(torch.topk(similarity, how_many, dim=-1).indices)

        return torch.cat(found, dim=0)

    @staticmethod
    def _normalise_neighbour_graph(indices: Tensor, side: int) -> Tensor:
        """Normalise one modality's neighbour graph so a hop preserves scale.

        Args:
            indices (Tensor): The (row, column) pairs of the neighbour graph.
            side (int): The side of the square matrix.

        Returns:
            Tensor: The normalised sparse adjacency.
        """
        ones = torch.ones(indices.size(1))
        degree = torch.zeros(side).index_add_(0, indices[0], ones) + 1e-7

        inverse = degree.pow(-0.5)
        values = inverse[indices[0]] * inverse[indices[1]]

        return torch.sparse_coo_tensor(indices, values, (side, side)).coalesce()

    def modality(self, name: str) -> Tensor:
        """One modality's feature table.

        Args:
            name (str): The modality to read.

        Returns:
            Tensor: The {(item + padding) x feature} table.
        """
        return getattr(self, f"modality_{name}")

    def modality_tables(self) -> List[Tensor]:
        """Every modality this model reads, in configuration order.

        Returns:
            List[Tensor]: One feature table per modality.
        """
        return [self.modality(name) for name in self.modality_names]

    def joint_features(self) -> Tensor:
        """Every modality laid side by side as one vector per item.

        Concatenating is what makes a single-feature model such as VBPR read
        several modalities without changing what it computes: a dot product
        against the concatenation is the sum of the dot products against each
        block, which is the additive form the paper writes.

        Returns:
            Tensor: The {(item + padding) x sum of widths} table.
        """
        tables = self.modality_tables()
        if len(tables) == 1:
            return tables[0]
        return torch.cat(tables, dim=1)
