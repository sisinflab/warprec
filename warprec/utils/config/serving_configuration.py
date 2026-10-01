import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

API_KEY_ENV = "WARPREC_API_KEY"

# The ingress deployment is called this, so no endpoint may be.
RESERVED_NAMES = {"gateway"}


class ServerSettings(BaseModel):
    """Where and how the serving application listens.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        host (str): The address the HTTP proxy binds to.
        port (int): The port the HTTP proxy listens on.
        route_prefix (str): The path every route is mounted under.
        api_key (Optional[str]): The key required in the X-API-Key header.
            None disables authentication. WARPREC_API_KEY overrides it.
        mcp (bool): Whether to expose the models as MCP tools under /mcp.
        ray_address (Optional[str]): The Ray cluster to join. None starts a
            local one; 'auto' joins the cluster this machine belongs to.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    route_prefix: str = "/"
    api_key: Optional[str] = None
    mcp: bool = False
    ray_address: Optional[str] = None

    @field_validator("route_prefix")
    @classmethod
    def check_route_prefix(cls, value: str) -> str:
        """Ray Serve mounts applications on absolute paths only.

        Args:
            value (str): The configured prefix.

        Returns:
            str: The prefix, unchanged.

        Raises:
            ValueError: If it does not start with a slash.
        """
        if not value.startswith("/"):
            raise ValueError(f"route_prefix must start with '/', got '{value}'.")
        return value

    def effective_api_key(self) -> Optional[str]:
        """The key in force: the environment wins over the file.

        Returns:
            Optional[str]: The key, or None when authentication is off.
        """
        return os.environ.get(API_KEY_ENV) or self.api_key


class BatchingSettings(BaseModel):
    """How concurrent requests are grouped into one forward pass.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        max_batch_size (int): The most requests scored together.
        batch_wait_timeout_s (float): How long the first request of a batch
            waits for others to join it.
    """

    model_config = ConfigDict(extra="forbid")

    max_batch_size: int = Field(64, ge=1)
    batch_wait_timeout_s: float = Field(0.005, ge=0)


class DeploymentSettings(BaseModel):
    """The Ray Serve options of one endpoint, passed through unchanged.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        num_replicas (Optional[Union[int, Literal["auto"]]]): A fixed replica
            count, or 'auto' for Ray's default autoscaling.
        max_ongoing_requests (Optional[int]): The most requests one replica
            handles at once.
        autoscaling_config (Optional[Dict[str, Any]]): Ray Serve autoscaling
            settings, such as min_replicas, max_replicas and
            target_ongoing_requests.
        ray_actor_options (Optional[Dict[str, Any]]): Resources per replica,
            such as num_cpus and num_gpus.
    """

    model_config = ConfigDict(extra="forbid")

    num_replicas: Optional[Union[int, Literal["auto"]]] = None
    max_ongoing_requests: Optional[int] = Field(None, ge=1)
    autoscaling_config: Optional[Dict[str, Any]] = None
    ray_actor_options: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_scaling(self) -> "DeploymentSettings":
        """A fixed replica count and an autoscaling policy contradict each other.

        Returns:
            DeploymentSettings: The validated settings.

        Raises:
            ValueError: If both a fixed num_replicas and autoscaling_config are set,
                or num_replicas is below one.
        """
        if isinstance(self.num_replicas, int):
            if self.num_replicas < 1:
                raise ValueError("num_replicas must be at least 1.")
            if self.autoscaling_config is not None:
                raise ValueError(
                    "Set either a fixed num_replicas or autoscaling_config, not both."
                )
        return self


class ItemMetadata(BaseModel):
    """A file that names the items, so responses and MCP tools can use titles.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        path (str): The delimited file with one row per item.
        sep (str): The column separator.
        header (bool): Whether the first row holds column names.
        id_column (Union[int, str]): The column of item ids, by position or name.
        name_column (Union[int, str]): The column of item names, by position or name.
        encoding (str): The file encoding.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    sep: str = ","
    header: bool = True
    id_column: Union[int, str] = 0
    name_column: Union[int, str] = 1
    encoding: str = "utf-8"

    @field_validator("path")
    @classmethod
    def check_path(cls, value: str) -> str:
        """The file has to exist where the server starts.

        Args:
            value (str): The configured path.

        Returns:
            str: The path, unchanged.

        Raises:
            ValueError: If there is no file at that path.
        """
        if not Path(value).is_file():
            raise ValueError(f"Item metadata file '{value}' does not exist.")
        return value


class EndpointConfig(BaseModel):
    """One served model.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        name (str): The name the model is served under, used in its URL.
        checkpoint (str): The .pth file written with meta.save_model.
        device (str): Where the model runs: cpu, mps, cuda or cuda:N.
        default_k (int): How many items a request gets when it does not say.
        max_k (int): The most items a request may ask for.
        mask_seen (bool): Whether items a user saw in training are left out.
        unknown_user (Literal["error", "popular"]): What a user unseen in
            training gets: a 404, or the most popular items.
        item_metadata (Optional[ItemMetadata]): Item names, when available.
        batching (BatchingSettings): How requests are grouped.
        deployment (DeploymentSettings): Ray Serve replica and resource options.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    checkpoint: str
    device: str = Field("cpu", pattern=r"^(cpu|mps|cuda(:\d+)?)$")
    default_k: int = Field(10, ge=1)
    max_k: int = Field(100, ge=1)
    mask_seen: bool = True
    unknown_user: Literal["error", "popular"] = "error"
    item_metadata: Optional[ItemMetadata] = None
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    deployment: DeploymentSettings = Field(
        default_factory=lambda: DeploymentSettings(num_replicas=1)
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        """The ingress deployment's name cannot be taken by a model.

        Args:
            value (str): The configured name.

        Returns:
            str: The name, unchanged.

        Raises:
            ValueError: If the name is reserved.
        """
        if value.lower() in RESERVED_NAMES:
            raise ValueError(f"'{value}' is a reserved name.")
        return value

    @field_validator("checkpoint")
    @classmethod
    def check_checkpoint(cls, value: str) -> str:
        """The checkpoint has to exist wherever the configuration is loaded.

        Args:
            value (str): The configured path.

        Returns:
            str: The path, unchanged.

        Raises:
            ValueError: If there is no file at that path.
        """
        if not Path(value).is_file():
            raise ValueError(f"Checkpoint '{value}' does not exist.")
        return value

    @model_validator(mode="after")
    def check_k(self) -> "EndpointConfig":
        """The default list length has to be one a request could ask for.

        Returns:
            EndpointConfig: The validated endpoint.

        Raises:
            ValueError: If default_k exceeds max_k.
        """
        if self.default_k > self.max_k:
            raise ValueError(
                f"default_k ({self.default_k}) cannot exceed max_k ({self.max_k})."
            )
        return self

    def deployment_options(self) -> Dict[str, Any]:
        """The options handed to Ray Serve for this endpoint's deployment.

        A replica is given no GPU unless it asks for one, and CUDA then sees no
        device at all, so an endpoint placed on cuda asks for one whole GPU
        unless its configuration already says how much it needs.

        Returns:
            Dict[str, Any]: The keyword arguments for Deployment.options().
        """
        options = self.deployment.model_dump(exclude_none=True)
        if self.device.startswith("cuda"):
            actor = dict(options.get("ray_actor_options") or {})
            actor.setdefault("num_gpus", 1)
            options["ray_actor_options"] = actor
        return options


class ServingConfiguration(BaseModel):
    """The configuration of a WarpRec serving application.

    Attributes:
        model_config: Configuration of the PyDantic model; unknown keys are rejected.
        server (ServerSettings): Where and how it listens.
        endpoints (List[EndpointConfig]): The models it serves.
    """

    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    endpoints: List[EndpointConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_names(self) -> "ServingConfiguration":
        """Each endpoint is addressed by name, so no two may share one.

        Returns:
            ServingConfiguration: The validated configuration.

        Raises:
            ValueError: If a name repeats.
        """
        names = [endpoint.name for endpoint in self.endpoints]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Endpoint names must be unique, repeated: {duplicates}.")
        return self

    def portable(self) -> "ServingConfiguration":
        """A copy that can be shipped to a cluster: absolute paths, no secret.

        Returns:
            ServingConfiguration: The copy.
        """
        copy = self.model_copy(deep=True)
        copy.server.api_key = None
        for endpoint in copy.endpoints:
            endpoint.checkpoint = str(Path(endpoint.checkpoint).resolve())
            if endpoint.item_metadata is not None:
                endpoint.item_metadata.path = str(
                    Path(endpoint.item_metadata.path).resolve()
                )
        return copy


def load_serving_configuration(path: Union[str, Path]) -> ServingConfiguration:
    """Read and validate a serving configuration file.

    Args:
        path (Union[str, Path]): The YAML file.

    Returns:
        ServingConfiguration: The validated configuration.
    """
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return ServingConfiguration.model_validate(data)
