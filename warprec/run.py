import argparse
from argparse import Namespace
from pathlib import Path

from warprec.pipelines import design_pipeline, train_pipeline, eval_pipeline


def main(args: Namespace):
    path = args.config
    pipeline = args.pipeline.lower()

    if not Path(path).is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    if pipeline not in ["train", "design", "eval"]:
        raise ValueError(
            f"Invalid pipeline specified: {pipeline}. Choose 'train', 'design' or 'eval'."
        )

    match pipeline:
        case "train":
            train_pipeline(path)
        case "design":
            design_pipeline(path)
        case "eval":
            eval_pipeline(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration file argument
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="store",
        required=True,
        help="Config file local path",
    )

    # Pipeline selection argument
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        action="store",
        required=True,
        help="Pipeline to run (train, design)",
    )

    # Parse arguments and execute main function
    args = parser.parse_args()
    main(args)
