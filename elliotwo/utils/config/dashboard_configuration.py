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
        codecarbon (CodeCarbon): Configuration for CodeCarbon.
    """

    wandb: Wandb = Field(default_factory=Wandb)
    codecarbon: CodeCarbon = Field(default_factory=CodeCarbon)
