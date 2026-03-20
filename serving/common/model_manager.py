"""Model and dataset lifecycle management for WarpRec serving.

Handles loading model checkpoints and dataset files at startup, building the
internal item-name-to-index mappings, and providing retrieval methods used by
the inference layer.
"""

import os
import re
from typing import Dict, List, Optional

import pandas as pd
import torch

from warprec.recommenders.base_recommender import Recommender
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry

from .config import ServingConfig


class ModelManager:
    """Loads and stores model-dataset pairs declared in the serving configuration.

    After calling ``load_all()``, models and their associated dataset mappings
    are available via ``get_model()`` using the
    ``"{model}_{dataset}"`` key format.

    Args:
        config (ServingConfig): Parsed serving configuration.
    """

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._models: dict[str, Recommender] = {}
        self._endpoint_types: dict[str, str] = {}

        # Mapping: dataset_name -> {item_name: external_id}
        self._name_to_ext: dict[str, dict[str, int]] = {}
        # Mapping: dataset_name -> {external_id: item_name}
        self._ext_to_name: dict[str, dict[int, str]] = {}
        # Mapping: model_key -> dataset_name (to know which map to use)
        self._model_to_dataset: dict[str, str] = {}

    # -- public API ----------------------------------------------------------

    def load_all(self) -> None:
        """Load every model-dataset pair listed in ``config.endpoints``.

        For each endpoint entry the method:
        1. Validates that the checkpoint and dataset files exist on disk.
        2. Loads the dataset file and builds an item-name-to-external-id mapping.
        3. Loads the model checkpoint via the warprec model registry.
        4. Combines the external-id mapping with item name to produce a direct
            item-name-to-external-index lookup table.
        """
        checkpoints_dir = self._config.checkpoints_dir
        datasets_dir = self._config.datasets_dir

        # Load raw datasets and build bidirectional external mappings
        for ds in self._config.datasets:
            dataset_path = os.path.join(datasets_dir, ds.item_mapping)
            if not os.path.exists(dataset_path):
                logger.msg(f"Dataset file not found at {dataset_path}. Skipping.")
                continue

            df = pd.read_csv(
                dataset_path,
                sep=ds.separator,
                encoding="latin-1",
                engine="python",
                header=None,
            )

            n2e: dict[str, int] = {}
            e2n: dict[int, str] = {}

            for _, row in df.iterrows():
                ext_id = int(row.iloc[0])
                item_name = str(row.iloc[1])
                # Strip year suffix like " (1995)"
                item_name = re.sub(r" \(\d{4}\)$", "", item_name)

                n2e[item_name] = ext_id
                e2n[ext_id] = item_name

            self._name_to_ext[ds.name] = n2e
            self._ext_to_name[ds.name] = e2n

        # Load models and link them to datasets
        for ep in self._config.endpoints:
            checkpoint_path = os.path.join(
                checkpoints_dir, f"{ep.model}_{ep.dataset}.pth"
            )
            if (
                not os.path.exists(checkpoint_path)
                or ep.dataset not in self._name_to_ext
            ):
                continue

            checkpoint = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )
            model_cls = model_registry.get_class(checkpoint["name"])
            loaded_model: Recommender = model_cls.from_checkpoint(checkpoint=checkpoint)

            self._models[ep.key] = loaded_model.to(ep.device)
            self._endpoint_types[ep.key] = ep.type
            self._model_to_dataset[ep.key] = ep.dataset

            logger.msg(f"Loaded endpoint '{ep.key}' linked to dataset '{ep.dataset}'.")

    def get_model(self, model_key: str) -> Recommender:
        """Retrieve a loaded model by its key.

        Args:
            model_key (str): Identifier in ``"{model}_{dataset}"`` format.

        Returns:
            Recommender: The loaded recommender model instance.

        Raises:
            KeyError: If the model key is not available.
        """
        if model_key not in self._models:
            available = ", ".join(self._models) or "(none)"
            raise KeyError(f"Model '{model_key}' is not loaded. Available: {available}")
        return self._models[model_key]

    def get_endpoint_type(self, model_key: str) -> str:
        """Return the recommender type for a given model key.

        Args:
            model_key (str): Identifier in ``"{model}_{dataset}"`` format.

        Returns:
            str: One of ``"sequential"``, ``"collaborative"``, or ``"contextual"``.

        Raises:
            KeyError: If the model key is not available.
        """
        if model_key not in self._endpoint_types:
            raise KeyError(f"Model '{model_key}' is not loaded.")
        return self._endpoint_types[model_key]

    def list_available_keys(self) -> List[str]:
        """Return all loaded model-dataset keys."""
        return list(self._models.keys())

    def get_available_endpoints(self) -> Dict[str, str]:
        """Return a mapping of model keys to their recommender types."""
        return dict(self._endpoint_types)

    def get_dataset_for_model(self, model_key: str) -> str:
        """Return the dataset name associated with a specific model key."""
        return self._model_to_dataset.get(model_key)

    def name_to_external_id(self, dataset_name: str, item_name: str) -> Optional[int]:
        """Convert an item name to its external ID."""
        return self._name_to_ext.get(dataset_name, {}).get(item_name)

    def external_id_to_name(self, dataset_name: str, ext_id: int) -> str:
        """Convert an external ID back to its item name."""
        return self._ext_to_name.get(dataset_name, {}).get(
            ext_id, f"Unknown_ID_{ext_id}"
        )
