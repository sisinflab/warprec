from pathlib import Path
from typing import Optional

import csv

from warprec.data.writer.base_writer import Writer
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.logger import logger


class LocalWriter(Writer):
    """LocalWriter saves the experiment results to the local filesystem.

    Args:
        dataset_name (str): The name of the dataset.
        local_path (str): The root path for saving experiments.
    """

    def __init__(self, dataset_name: str, local_path: str):
        super().__init__()

        # Setup
        self.experiment_path = Path(self._path_join(local_path, dataset_name))
        self.experiment_evaluation_path = Path(
            self._path_join(self.experiment_path, "evaluation")
        )
        self.experiment_recommendation_path = Path(
            self._path_join(self.experiment_path, "recs")
        )
        self.experiment_serialized_models_path = Path(
            self._path_join(self.experiment_path, "serialized")
        )
        self.experiment_params_path = Path(
            self._path_join(self.experiment_path, "params")
        )
        self.experiment_split_path = Path(
            self._path_join(self.experiment_path, "split")
        )

        self.setup_experiment()

    def _path_join(self, *args) -> str:
        """Joins path components for the local filesystem."""
        return str(Path(args[0]).joinpath(*args[1:]))

    def setup_experiment(self):
        """Creates all necessary local directories for the experiment."""
        logger.msg("Setting up experiment local folder.")
        for path in [
            self.experiment_path,
            self.experiment_evaluation_path,
            self.experiment_recommendation_path,
            self.experiment_serialized_models_path,
            self.experiment_params_path,
            self.experiment_split_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        logger.msg("Experiment folder created successfully.")

    def write_recs(
        self,
        model: Recommender,
        dataset: Dataset,
        k: int,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        user_label: str = "user_id",
        item_label: str = "item_id",
        rating_label: str = "rating",
        chunk_size: int = 100_000,
    ):
        """Writes recommendations to a local file in a streaming fashion."""
        path = self._path_join(
            self.experiment_recommendation_path,
            f"{model.name}_{self._timestamp}{ext}",
        )

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=sep)
                if header:
                    writer.writerow([user_label, item_label, rating_label])

                # Create the row generator and chunk generator
                row_generator = self._generate_recommendation_rows(model, dataset, k)
                chunk_generator = self._chunk_generator(row_generator, chunk_size)

                # Iterate over chunks and write to file
                for chunk in chunk_generator:
                    writer.writerows(chunk)
            logger.msg(f"Recommendations successfully written to {path}")
        except Exception as e:
            logger.negative(f"Error writing recommendations to {path}: {e}")

    def _write_text(self, path: str, content: str) -> None:
        """Writes text content to a local file."""
        Path(path).write_text(content, encoding="utf-8")

    def _read_text(self, path: str) -> Optional[str]:
        """Reads text content from a local file if it exists."""
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            return p.read_text(encoding="utf-8")
        return None

    def _write_bytes(self, path: str, content: bytes) -> None:
        """Writes binary content to a local file."""
        Path(path).write_bytes(content)
