import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

from warprec.errors import ConfigurationError, DataNotFoundError, WarpRecError

from warprec.pipelines import (
    design_pipeline,
    estimate_pipeline,
    train_pipeline,
    eval_pipeline,
    swarm_pipeline,
)

# Disable warnings during pipelines execution
warnings.filterwarnings("ignore", category=UserWarning)


def main(argv: Optional[List[str]] = None) -> None:
    """Main Warprec access point.

    This function will parse and pass the correct arguments
    to the pre-constructed Warprec pipelines.

    A WarpRecError, an error the user can fix, is printed to stderr as
    '<ClassName>: <message>' without a traceback, and the process exits with
    status 1; '--debug' raises it instead so the traceback shows. Any other
    exception is raised as is.

    Args:
        argv (Optional[List[str]]): The command-line arguments; None reads sys.argv.

    Raises:
        SystemExit: With status 1 when a WarpRecError stops the run.
        WarpRecError: When a WarpRecError stops the run under '--debug'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Config file local path"
    )
    parser.add_argument(
        "-p", "--pipeline", type=str, required=True, help="Pipeline to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the traceback of a configuration or data error",
    )

    args = parser.parse_args(argv)

    try:
        _run(args.config, args.pipeline.lower())
    except WarpRecError as error:
        if args.debug:
            raise
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


def _run(path: str, pipeline: str) -> None:
    """Run a pipeline on a configuration file.

    Args:
        path (str): The configuration file.
        pipeline (str): The pipeline's name, in lower case.

    Raises:
        DataNotFoundError: If the configuration file cannot be found.
        ConfigurationError: If the pipeline selected is not supported.
    """
    if not Path(path).is_file():
        raise DataNotFoundError(f"Configuration file not found at: {path}")

    if pipeline not in [
        "train",
        "design",
        "eval",
        "swarm",
        "estimate",
    ]:
        raise ConfigurationError(
            "Invalid pipeline specified: "
            f"{pipeline}. Choose 'train', 'design', 'eval', 'swarm' or 'estimate'."
        )

    match pipeline:
        case "train":
            train_pipeline(path)
        case "design":
            design_pipeline(path)
        case "eval":
            eval_pipeline(path)
        case "swarm":
            swarm_pipeline(path)
        case "estimate":
            estimate_pipeline(path)


if __name__ == "__main__":
    main()
