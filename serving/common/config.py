"""Configuration classes for the WarpRec serving module.

Loads settings from a YAML configuration file with environment variable overrides
for sensitive values like API keys. Provides a single source of truth for both
the REST API and MCP server applications.
"""

import os
from pathlib import Path
from typing import Literal, List, Optional

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Metadata for a dataset file used by one or more endpoints.

    Attributes:
        name (str): Identifier for the dataset (e.g., "movielens").
        item_mapping (str): Filename of the item mapping file inside the datasets directory
            (e.g., "movies.dat").
        separator (str): Column separator used in the file (e.g., "::", ",", "\\t").
    """

    name: str
    item_mapping: str
    separator: str = "::"


ModelType = Literal["sequential", "collaborative", "contextual"]


class EndpointConfig(BaseModel):
    """A single model-dataset combination to load at startup.

    Attributes:
        model (str): Name of the recommender model (e.g., "SASRec", "BPR").
        dataset (str): Name of the dataset (must match a name in the datasets section).
        type (ModelType): Recommender category that determines the inference flow.
        device (str): PyTorch device for this endpoint (e.g., "cpu", "cuda:0").
    """

    model: str
    dataset: str
    type: ModelType
    device: str = "cpu"

    @property
    def key(self) -> str:
        """Unique identifier for this model-dataset pair."""
        return f"{self.model}_{self.dataset}"


class ServerConfig(BaseModel):
    """Network configuration for the serving applications.

    Attributes:
        host (str): Bind address for the servers.
        rest_port (int): Port for the FastAPI REST server.
        mcp_port (int): Port for the MCP server.
        api_key (str): API key for REST API authentication (empty string disables auth).
    """

    host: str = "127.0.0.1"
    rest_port: int = 8080
    mcp_port: int = 8081
    api_key: str = ""


class ServingConfig(BaseModel):
    """Top-level configuration for the WarpRec serving module.

    Attributes:
        server (ServerConfig): Network settings for REST and MCP servers.
        checkpoints_dir (str): Directory containing model checkpoint files (.pth).
        datasets_dir (str): Directory containing dataset files.
        datasets (List[DatasetConfig]): List of dataset definitions with file metadata.
        endpoints (List[EndpointConfig]): List of model-dataset pairs to load at startup.
    """

    server: ServerConfig = ServerConfig()
    checkpoints_dir: str = "checkpoints"
    datasets_dir: str = "datasets"
    datasets: List[DatasetConfig] = []
    endpoints: List[EndpointConfig] = []

    def get_dataset_config(self, name: str) -> DatasetConfig:
        """Look up a dataset configuration by name.

        Args:
            name (str): The dataset identifier to search for.

        Returns:
            DatasetConfig: The matching DatasetConfig instance.

        Raises:
            KeyError: If no dataset with the given name is defined.
        """
        for ds in self.datasets:
            if ds.name == name:
                return ds
        available = ", ".join(ds.name for ds in self.datasets) or "(none)"
        raise KeyError(
            f"Dataset '{name}' is not defined in the configuration. "
            f"Available: {available}"
        )

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "ServingConfig":
        """Load configuration from a YAML file with environment variable overrides.

        The method looks for the config file in the following order:
        1. The explicit ``path`` argument.
        2. The ``SERVING_CONFIG_PATH`` environment variable.
        3. ``serving_config.yml`` next to this file's parent directory.

        Environment variable overrides (applied after YAML loading):
        - ``SERVER_HOST`` overrides ``server.host``
        - ``SERVER_API_KEY`` overrides ``server.api_key``

        Args:
            path (Optional[str]): Explicit path to the YAML configuration file.

        Returns:
            ServingConfig: A fully resolved ServingConfig instance.
        """
        if path is None:
            path = os.environ.get(
                "SERVING_CONFIG_PATH",
                str(Path(__file__).parent.parent / "serving_config.yml"),
            )

        config_path = Path(path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        config = cls.model_validate(raw)

        # Apply environment variable overrides for sensitive or deployment-specific values
        env_host = os.environ.get("SERVER_HOST")
        if env_host is not None:
            config.server.host = env_host

        env_api_key = os.environ.get("SERVER_API_KEY")
        if env_api_key is not None:
            config.server.api_key = env_api_key

        return config
