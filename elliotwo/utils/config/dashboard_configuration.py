from typing import Optional

from pydantic import BaseModel, Field


class Wandb(BaseModel):
    """Definition of Wandb configuration.

    Attributes:
        enabled (bool): Whether to enable Wandb tracking.
        project_name (Optional[str]): Name of the Wandb project.
        log_config (bool): Whether to log the configuration.
        upload_checkpoints (bool): Whether to upload checkpoints.
        save_checkpoints (bool): Whether to save checkpoints.
    """

    enabled: bool = False
    project_name: Optional[str] = None
    log_config: bool = False
    upload_checkpoints: bool = False
    save_checkpoints: bool = False


class MLflow(BaseModel):
    """Definition of MLflow configuration.

    Attributes:
        enabled (bool): Whether to enable MLflow tracking.
        tracking_uri (str): URI of the MLflow tracking server.
        registry_uri (str): URI of the MLflow model registry.
        experiment_name (Optional[str]): Name of the MLflow experiment.
        tags (dict): Tags to be added to the MLflow run.
        tracking_token (Optional[str]): Token for MLflow tracking.
        save_artifacts (bool): Whether to save the artifacts.
    """

    enabled: bool = False
    tracking_uri: str = "mlruns/"
    registry_uri: str = "mlruns/"
    experiment_name: Optional[str] = None
    tags: dict = {}
    tracking_token: Optional[str] = None
    save_artifacts: bool = False


class CodeCarbon(BaseModel):
    """Definition of CodeCarbon configuration.

    Attributes:
        enabled (bool): Whether to enable CodeCarbon tracking.
        save_to_api (bool): Whether to save the results to CodeCarbon API.
        save_to_file (bool): Whether to save the results to a file.
        output_dir (str): Directory to save the results.
        tracking_mode (str): Tracking mode for CodeCarbon.
            Options are "machine" or "process".
    """

    enabled: bool = False
    save_to_api: bool = False
    save_to_file: bool = False
    output_dir: str = "./"
    tracking_mode: str = "machine"


class DashboardConfig(BaseModel):
    """Definition of Dashboard configuration.

    Attributes:
        wandb (Wandb): Configuration for Weights and Biases.
        mlflow (MLflow): Configuration for MLflow.
        codecarbon (CodeCarbon): Configuration for CodeCarbon.
    """

    wandb: Wandb = Field(default_factory=Wandb)
    mlflow: MLflow = Field(default_factory=MLflow)
    codecarbon: CodeCarbon = Field(default_factory=CodeCarbon)
