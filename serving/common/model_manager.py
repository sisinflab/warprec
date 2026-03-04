"""Model and dataset lifecycle management for WarpRec serving.

Handles loading model checkpoints and dataset files at startup, building the
internal item-name-to-index mappings, and providing retrieval methods used by
the inference layer.
"""

import os
import re

import pandas as pd
import torch

from warprec.recommenders.base_recommender import Recommender
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry

from .config import ServingConfig


class ModelManager:
    """Loads and stores model-dataset pairs declared in the serving configuration.

    After calling ``load_all()``, models and their associated dataset mappings
    are available via ``get_model()`` and ``get_dataset_mapping()`` using the
    ``"{model}_{dataset}"`` key format.

    Args:
        config: Parsed serving configuration.
    """

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._models: dict[str, Recommender] = {}
        self._dataset_mappings: dict[str, dict[str, int]] = {}
        self._endpoint_types: dict[str, str] = {}

    # -- public API ----------------------------------------------------------

    def load_all(self) -> None:
        """Load every model-dataset pair listed in ``config.endpoints``.

        For each endpoint entry the method:
        1. Validates that the checkpoint and dataset files exist on disk.
        2. Loads the dataset file and builds an item-name-to-external-id mapping.
        3. Loads the model checkpoint via the warprec model registry.
        4. Combines the external-id mapping with the model's internal item mapping
           to produce a direct item-name-to-internal-index lookup table.
        """
        checkpoints_dir = self._config.checkpoints_dir
        datasets_dir = self._config.datasets_dir

        # Load raw dataset files using metadata from the datasets section.
        # Each DatasetConfig provides the item mapping filename and separator.
        raw_datasets: dict[str, pd.DataFrame] = {}
        for ds in self._config.datasets:
            dataset_path = os.path.join(datasets_dir, ds.item_mapping)
            if not os.path.exists(dataset_path):
                logger.msg(
                    f"Dataset file not found at {dataset_path}. "
                    f"Endpoints using '{ds.name}' will be skipped."
                )
                continue
            raw_datasets[ds.name] = pd.read_csv(
                dataset_path,
                sep=ds.separator,
                encoding="latin-1",
                engine="python",
                header=None,
            )

        # Build item-name-to-external-id mapping for each loaded dataset.
        # Column 0 is external id, column 1 is item name.
        # Year suffixes like " (1995)" are stripped from item names.
        item_to_ext: dict[str, dict[str, int]] = {}
        for name, df in raw_datasets.items():
            mapping: dict[str, int] = {}
            for _, row in df.iterrows():
                item_name = str(row.iloc[1])
                item_name = re.sub(r" \(\d{4}\)$", "", item_name)
                mapping[item_name] = row.iloc[0]
            item_to_ext[name] = mapping

        # Load each endpoint's model checkpoint and build the final mapping
        for ep in self._config.endpoints:
            checkpoint_path = os.path.join(
                checkpoints_dir, f"{ep.model}_{ep.dataset}.pth"
            )
            if not os.path.exists(checkpoint_path):
                logger.msg(
                    f"Checkpoint not found at {checkpoint_path}. "
                    f"Endpoint '{ep.key}' will be skipped."
                )
                continue

            if ep.dataset not in raw_datasets:
                logger.msg(
                    f"Dataset '{ep.dataset}' was not loaded. "
                    f"Endpoint '{ep.key}' will be skipped."
                )
                continue

            # Load and instantiate the model from its checkpoint
            checkpoint = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )
            model_cls = model_registry.get_class(checkpoint["name"])
            loaded_model: Recommender = model_cls.from_checkpoint(checkpoint=checkpoint)
            loaded_model = loaded_model.to(ep.device)

            self._models[ep.key] = loaded_model
            self._endpoint_types[ep.key] = ep.type

            # Build item-name -> internal-index mapping by chaining
            # item-name -> external-id -> internal-index (from model.info["item_mapping"])
            model_item_mapping = loaded_model.info.get("item_mapping", {})
            ext_mapping = item_to_ext.get(ep.dataset, {})
            final_mapping: dict[str, int] = {}
            for item_name, ext_id in ext_mapping.items():
                if ext_id in model_item_mapping:
                    final_mapping[item_name] = model_item_mapping[ext_id]
            self._dataset_mappings[ep.dataset] = final_mapping

            logger.msg(
                f"Loaded endpoint '{ep.key}' ({ep.type}) on device '{ep.device}'."
            )

    def get_model(self, model_key: str) -> Recommender:
        """Retrieve a loaded model by its key.

        Args:
            model_key: Identifier in ``"{model}_{dataset}"`` format.

        Returns:
            The loaded recommender model instance.

        Raises:
            KeyError: If the model key is not available.
        """
        if model_key not in self._models:
            available = ", ".join(self._models) or "(none)"
            raise KeyError(f"Model '{model_key}' is not loaded. Available: {available}")
        return self._models[model_key]

    def get_dataset_mapping(self, dataset_name: str) -> dict[str, int]:
        """Retrieve the item-name-to-internal-index mapping for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "movielens").

        Returns:
            Dictionary mapping item names to internal model indices.

        Raises:
            KeyError: If the dataset mapping is not available.
        """
        if dataset_name not in self._dataset_mappings:
            available = ", ".join(self._dataset_mappings) or "(none)"
            raise KeyError(
                f"Dataset mapping for '{dataset_name}' is not available. "
                f"Available: {available}"
            )
        return self._dataset_mappings[dataset_name]

    def get_endpoint_type(self, model_key: str) -> str:
        """Return the recommender type for a given model key.

        Args:
            model_key: Identifier in ``"{model}_{dataset}"`` format.

        Returns:
            One of ``"sequential"``, ``"collaborative"``, or ``"contextual"``.

        Raises:
            KeyError: If the model key is not available.
        """
        if model_key not in self._endpoint_types:
            raise KeyError(f"Model '{model_key}' is not loaded.")
        return self._endpoint_types[model_key]

    def list_available_keys(self) -> list[str]:
        """Return all loaded model-dataset keys."""
        return list(self._models.keys())

    def get_available_endpoints(self) -> dict[str, str]:
        """Return a mapping of model keys to their recommender types."""
        return dict(self._endpoint_types)
