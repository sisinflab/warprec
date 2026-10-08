import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import narwhals as nw
import pandas as pd
import torch

from warprec.utils.callback import WarpRecCallback


class GuideCallback(WarpRecCallback):
    """A callback that uses every WarpRec hook and one Ray Tune hook.

    It drops the ratings dated before the film's release, puts each film's release year in the
    stash of every dataset, records the validation score of every trial report,
    and writes the test results of every model to a file. Each hook also appends
    a line to events.jsonl naming the process it ran in, which is how the guide
    shows where WarpRec calls it.

    The callback is pickled to reach Ray workers (the data preparation and every
    trial of the train pipeline), so it holds paths and small lists only, never
    the data itself.

    Args:
        *args (Any): The configuration's 'args', unused here.
        items_path (str): The items file with an item_id and a release_date column.
        output_dir (str): Where the callback writes its files.
        metric (str): The validation score recorded from every trial report.
        **kwargs (Any): The rest of the configuration's 'kwargs', unused here.
    """

    def __init__(
        self,
        *args: Any,
        items_path: str = "items.tsv",
        output_dir: str = ".",
        metric: str = "nDCG@10",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        # Resolved here, in the process that reads the configuration, so that a
        # Ray worker started in another directory reads the same files.
        self.items_path = Path(items_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rows_kept: Optional[int] = None
        self.trial_reports: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        self._event("__init__")

    def _event(self, hook: str, **details: Any):
        """Append one line to events.jsonl: the hook, the process and details."""
        line = {"hook": hook, "pid": os.getpid(), **details}
        with open(self.output_dir / "events.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")

    def __setstate__(self, state: Dict[str, Any]):
        """Record every time a copy of the callback is rebuilt from a pickle."""
        self.__dict__.update(state)
        self._event("unpickled")

    def _release_dates(self) -> pd.DataFrame:
        """Read the items file: item_id, and release date as a Unix timestamp."""
        items = pd.read_csv(self.items_path, sep="\t")
        released = pd.to_datetime(
            items.release_date, format="%d-%b-%Y", errors="coerce"
        )
        return pd.DataFrame({"item_id": items.item_id, "released": released})

    def on_data_reading(self, data):
        """Drop the ratings dated before the film's release date.

        The check needs a file WarpRec does not read, so no filter can do it.
        Films without a release date keep all their ratings.
        """
        dates = self._release_dates().dropna()
        released = nw.from_dict(
            {
                "item_id": dates.item_id.tolist(),
                "released": (dates.released.astype("int64") // 10**9).tolist(),
            },
            backend=nw.get_native_namespace(data),
        ).with_columns(nw.col("item_id").cast(data.schema["item_id"]))

        kept = (
            data.join(released, on="item_id", how="left")
            .filter(
                nw.col("released").is_null()
                | (nw.col("timestamp") >= nw.col("released"))
            )
            .drop("released")
        )
        self.rows_kept = len(kept)
        self._event("on_data_reading", rows_read=len(data), rows_kept=len(kept))
        return kept

    def on_dataset_creation(
        self, main_dataset, val_dataset, validation_folds, *args, **kwargs
    ):
        """Put the release year of every item in the stash of every dataset.

        The tensor is indexed like the dataset's items, through its own item
        mapping, with one more slot for the padding index. Films without a
        release date get NaN.
        """
        dates = self._release_dates()
        year_by_id = dict(zip(dates.item_id, dates.released.dt.year))

        datasets = [main_dataset, val_dataset, *validation_folds]
        datasets = [dataset for dataset in datasets if dataset is not None]
        for dataset in datasets:
            _, item_map = dataset.get_mappings()
            item_year = torch.full((dataset.get_dims()[1] + 1,), float("nan"))
            for raw_id, index in item_map.items():
                item_year[index] = year_by_id.get(raw_id, float("nan"))
            dataset.add_to_stash("item_year", item_year)

        self._event("on_dataset_creation", datasets=len(datasets))

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Ray Tune hook: record the validation score and loss of every trial report."""
        self.trial_reports.append(
            {
                "trial": str(trial),
                "params": {
                    key: value
                    for key, value in trial.config.items()
                    if not isinstance(value, dict)
                },
                "report": result.get("training_iteration"),
                self.metric: result.get(self.metric),
                "train_loss": result.get("train_loss"),
            }
        )
        self._event("on_trial_result", trial=str(trial))

    def on_training_complete(self, model, *args, **kwargs):
        """Note which model finished training, and what this copy has seen."""
        self._event(
            "on_training_complete",
            model=model.name,
            rows_kept=self.rows_kept,
            trial_reports=len(self.trial_reports),
        )

    def on_evaluation_complete(self, model, params, results, *args, **kwargs):
        """Collect the test results and write them, with the trial reports."""
        for k, metrics in results.items():
            row = {"model": model.name, "k": k}
            for name, value in metrics.items():
                row[name] = value.nanmean().item() if torch.is_tensor(value) else value
            self.test_results.append(row)

        pd.DataFrame(self.test_results).to_csv(
            self.output_dir / "test_results.tsv", sep="\t", index=False
        )
        if self.trial_reports:
            pd.DataFrame(self.trial_reports).to_json(
                self.output_dir / "trial_reports.jsonl", orient="records", lines=True
            )
        self._event("on_evaluation_complete", model=model.name)
