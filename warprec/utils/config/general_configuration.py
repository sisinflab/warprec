from typing import Optional

from pydantic import BaseModel


class GeneralConfig(BaseModel):
    """Definition of the general configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        precision (Optional[str]): The precision to use during computation.
        batch_size (Optional[int]): The batch_size used during the experiment.
        ray_verbose (Optional[int]): The Ray level of verbosity.
    """

    precision: Optional[str] = "float32"
    batch_size: Optional[int] = 1024
    ray_verbose: Optional[int] = 1
