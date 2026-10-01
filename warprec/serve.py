import argparse
from pathlib import Path
from typing import List, Optional

from warprec.utils.config.serving_configuration import load_serving_configuration


def main(argv: Optional[List[str]] = None) -> None:
    """Serve trained WarpRec models, or write them out as a Ray Serve config.

    Args:
        argv (Optional[List[str]]): The command-line arguments; None reads sys.argv.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        SystemExit: If the serving extra is not installed.
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
    args = parser.parse_args(argv)

    if not Path(args.config).is_file():
        raise FileNotFoundError(f"Configuration file not found at: {args.config}")
    config = load_serving_configuration(args.config)

    try:
        # Imported late: Ray Serve is an optional extra.
        from warprec.serving.app import (  # pylint: disable=import-outside-toplevel
            export_serve_config,
            run,
        )
    except ImportError as error:
        raise SystemExit(
            f"Serving needs the 'serving' extra ({error}). "
            "Install it with: pip install 'warprec[serving]'"
        ) from error

    if args.export:
        export_serve_config(config, args.export)
        return
    run(config)


if __name__ == "__main__":
    main()
