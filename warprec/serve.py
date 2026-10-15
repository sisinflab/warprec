import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

from warprec.errors import DataNotFoundError, WarpRecError
from warprec.utils.config.serving_configuration import load_serving_configuration


def main(argv: Optional[List[str]] = None) -> None:
    """Serve trained WarpRec models, or write them out as a Ray Serve config.

    A WarpRecError, an error the user can fix, is printed to stderr as
    '<ClassName>: <message>' without a traceback, and the process exits with
    status 1; '--debug' raises it instead so the traceback shows. Any other
    exception is raised as is.

    Args:
        argv (Optional[List[str]]): The command-line arguments; None reads sys.argv.

    Raises:
        SystemExit: With status 1 when a WarpRecError stops the server.
        WarpRecError: When a WarpRecError stops the server under '--debug'.
    """
    parser = argparse.ArgumentParser(
        prog="warprec.serve",
        description="Serve trained WarpRec models with Ray Serve.",
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Serving config file path"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Write a Ray Serve config file instead of starting the server",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the traceback of a configuration or data error",
    )
    args = parser.parse_args(argv)

    try:
        _serve(args.config, args.export)
    except WarpRecError as error:
        if args.debug:
            raise
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


def _serve(path: str, export: Optional[str]) -> None:
    """Serve the models a configuration file names, or export the application.

    Args:
        path (str): The serving configuration file.
        export (Optional[str]): Where to write a Ray Serve config file instead
            of starting the server, or None to start it.

    Raises:
        DataNotFoundError: If the configuration file does not exist.
        SystemExit: If the serving extra, or the mcp extra when MCP is enabled,
            is not installed.
    """
    if not Path(path).is_file():
        raise DataNotFoundError(f"Configuration file not found at: {path}")
    config = load_serving_configuration(path)

    try:
        # Imported late: Ray Serve is an optional extra.
        from warprec.serving.app import (
            export_serve_config,
            run,
        )
    except ImportError as error:
        raise SystemExit(
            f"Serving needs the 'serving' extra ({error}). "
            "Install it with: pip install 'warprec[serving]'"
        ) from error

    if export:
        export_serve_config(config, export)
        return

    # Checked here rather than left to the gateway: inside Ray the missing
    # import only shows up as a deployment that never becomes ready.
    if config.server.mcp and importlib.util.find_spec("fastmcp") is None:
        raise SystemExit(
            "server.mcp is enabled but fastmcp is not installed. Install it with: "
            "pip install 'warprec[mcp]', or set server.mcp to false."
        )
    run(config)


if __name__ == "__main__":
    main()
