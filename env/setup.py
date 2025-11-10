import os
import yaml


def initialize_environment(
    path: str,
) -> None:
    """Initialize environment variable before starting execution.

    Args:
        path (str): The path to the configuration file.

    Returns:
        None: In case the configuration file does not exist.

    Raises:
        ValueError: When the argument of the cuda visible device field
            contains an invalid value.
        yaml.YAMLError: When the parsing of the YAML file raises an error.
        Exception: When an unknown error occurs.
    """

    # Check if configuration exists
    if not os.path.exists(path):
        return

    # Set Ray environment variable to enable new features
    os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

    try:
        with open(path, "r", encoding="utf-8") as file:
            # Read only the cuda devices part of the configuration
            config: dict = yaml.safe_load(file)
            visible_devices = config.get("general", {}).get(
                "cuda_visible_devices", None
            )

            # Nothing more to do
            if visible_devices is None:
                return

            # Validate the input
            if not isinstance(visible_devices, (int, list)):
                raise ValueError(
                    "Cuda visible devices must be an integer or a list of integers. "
                    f"Value received: {visible_devices}"
                )

            # Convert int to list
            if isinstance(visible_devices, int):
                visible_devices = [visible_devices]

            # Check that every item in list is an integer
            if any(not isinstance(device, int) for device in visible_devices):
                raise ValueError(
                    "Cuda visible devices must be an integer or a list of integers. "
                    f"Value received: {visible_devices}"
                )

            # Correctly set the environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_devices))

    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error during YAML file parsing: {exc}")
    except Exception as e:
        raise Exception(f"Unexpected error has occurred: {e}")
