import argparse
from pathlib import Path

from warprec.pipelines import design_pipeline, train_pipeline, eval_pipeline


def main():
    """Main Warprec access point.

    This function will parse and pass the correct arguments
    to the pre-constructed Warprec pipelines.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If the pipeline selected is not supported.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Config file local path"
    )
    parser.add_argument(
        "-p", "--pipeline", type=str, required=True, help="Pipeline to run"
    )

    args = parser.parse_args()

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
    main()
