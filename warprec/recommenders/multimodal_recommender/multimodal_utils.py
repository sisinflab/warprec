from typing import Any, List, Optional

import torch
from torch import Tensor, nn

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

    Raises:
        ValueError: If no features were provided, or a named modality is absent.
    """

    modality_names: List[str]
    modality_dims: List[int]

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
