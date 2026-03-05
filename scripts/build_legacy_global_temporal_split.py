#!/usr/bin/env python3
"""Build WarpRec-compatible split files from the legacy GlobalTemporalSplit logic."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/validation/test CSV files compatible with WarpRec, "
            "using the same timestamp cutoffs as the provided GlobalTemporalSplit "
            "and a sampled temporal validation tail inside the legacy train pool."
        )
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="data/lastfm1k_30000/lastfm_no_header.csv",
        help="Path to the interaction file. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lastfm1k_30000/legacy_global_split",
        help="Directory where train/validation/test files will be written.",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Column separator used by the dataset. Default: %(default)r",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row as header.",
    )
    parser.add_argument(
        "--write-header",
        action="store_true",
        help="Write a header row to the generated split files.",
    )
    parser.add_argument(
        "--user-column",
        type=int,
        default=0,
        help="Zero-based index of the user_id column. Default: %(default)s",
    )
    parser.add_argument(
        "--item-column",
        type=int,
        default=1,
        help="Zero-based index of the item_id column. Default: %(default)s",
    )
    parser.add_argument(
        "--rating-column",
        type=int,
        default=2,
        help="Zero-based index of the rating column. Default: %(default)s",
    )
    parser.add_argument(
        "--timestamp-column",
        type=int,
        default=3,
        help="Zero-based index of the timestamp column. Default: %(default)s",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction reserved for test in GlobalTemporalSplit. Default: %(default)s",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction reserved for validation on the pre-test partition, "
            "matching the provided GlobalTemporalSplit cutoff logic. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--max-test-users",
        type=int,
        default=992,
        help="Maximum number of test users. Default: %(default)s",
    )
    parser.add_argument(
        "--max-validation-users",
        type=int,
        default=128,
        help="Maximum number of validation users sampled from the legacy train pool.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=31337,
        help="Random seed used for deterministic user sampling. Default: %(default)s",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute and print statistics without writing split files.",
    )
    return parser.parse_args()


def validate_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in the open interval (0, 1), got {value}.")


def load_sorted_timestamps(
    dataset_path: Path,
    *,
    sep: str,
    timestamp_column: int,
    has_header: bool,
) -> list[str]:
    timestamps: list[str] = []

    with dataset_path.open("r", newline="") as source:
        reader = csv.reader(source, delimiter=sep)
        if has_header:
            next(reader, None)

        for row_number, row in enumerate(reader, start=2 if has_header else 1):
            if timestamp_column >= len(row):
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, "
                    f"but timestamp column {timestamp_column} was requested."
                )
            timestamps.append(row[timestamp_column])

    if not timestamps:
        raise ValueError("The dataset does not contain any timestamp values.")

    timestamps.sort()
    return timestamps


def compute_cutoffs(
    timestamps: list[str],
    *,
    test_fraction: float,
    validation_fraction: float,
) -> tuple[str, str]:
    total = len(timestamps)
    test_index = int(total * (1.0 - test_fraction))
    if test_index >= total:
        raise ValueError(
            f"Test index {test_index} is out of range for {total} timestamps. "
            "Decrease test_fraction."
        )

    test_timestamp = timestamps[test_index]
    pre_test_count = bisect.bisect_left(timestamps, test_timestamp)
    if pre_test_count == 0:
        raise ValueError("No timestamps fall before the computed test border timestamp.")

    validation_index = int(pre_test_count * (1.0 - validation_fraction))
    if validation_index >= total:
        raise ValueError(
            f"Validation index {validation_index} is out of range for {total} timestamps. "
            "Decrease validation_fraction."
        )

    validation_timestamp = timestamps[validation_index]
    return test_timestamp, validation_timestamp


def sample_users(user_ids: Iterable[str], max_users: int, seed: int) -> set[str]:
    ordered = sorted(set(user_ids))
    if len(ordered) <= max_users:
        return set(ordered)

    rng = random.Random(seed)
    rng.shuffle(ordered)
    return set(ordered[:max_users])


def sample_validation_users(user_ids: Iterable[str], max_users: int, seed: int) -> tuple[set[str], str]:
    ordered = sorted(set(user_ids))
    if len(ordered) <= max_users:
        return set(ordered), "all_users"

    try:
        import numpy as np  # type: ignore

        rng = np.random.RandomState(seed)
        chosen = rng.choice(ordered, max_users, replace=False).tolist()
        return set(chosen), "numpy_random_state"
    except ModuleNotFoundError:
        rng = random.Random(seed)
        return set(rng.sample(ordered, max_users)), "python_random_fallback"


def scan_dataset(
    dataset_path: Path,
    *,
    sep: str,
    has_header: bool,
    user_column: int,
    rating_column: int,
    timestamp_column: int,
    validation_timestamp: str,
    test_timestamp: str,
    workdir: Path,
) -> dict[str, object]:
    pre_validation_path = workdir / "pre_validation.csv"
    between_path = workdir / "between_validation_and_test.csv"
    post_test_path = workdir / "post_test.csv"

    positive_before_validation = defaultdict(int)
    users_before_test: set[str] = set()
    users_with_between: set[str] = set()
    positive_after_test: set[str] = set()

    with dataset_path.open("r", newline="") as source, pre_validation_path.open(
        "w", newline=""
    ) as pre_out, between_path.open("w", newline="") as between_out, post_test_path.open(
        "w", newline=""
    ) as post_out:
        reader = csv.reader(source, delimiter=sep)
        pre_writer = csv.writer(pre_out, delimiter=sep)
        between_writer = csv.writer(between_out, delimiter=sep)
        post_writer = csv.writer(post_out, delimiter=sep)

        if has_header:
            next(reader, None)

        for row_number, row in enumerate(reader, start=2 if has_header else 1):
            try:
                user_id = row[user_column]
                rating = float(row[rating_column])
                timestamp = row[timestamp_column]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid row at line {row_number}: {row!r}") from exc

            if timestamp < validation_timestamp:
                pre_writer.writerow(row)
                users_before_test.add(user_id)
                if rating > 0:
                    positive_before_validation[user_id] += 1
            elif timestamp < test_timestamp:
                between_writer.writerow(row)
                users_before_test.add(user_id)
                users_with_between.add(user_id)
            else:
                post_writer.writerow(row)
                if rating > 0:
                    positive_after_test.add(user_id)

    return {
        "pre_validation_path": pre_validation_path,
        "between_path": between_path,
        "post_test_path": post_test_path,
        "positive_before_validation": positive_before_validation,
        "users_before_test": users_before_test,
        "users_with_between": users_with_between,
        "positive_after_test": positive_after_test,
    }


def append_filtered_rows(
    input_path: Path,
    output_writer: csv.writer,
    *,
    sep: str,
    user_column: int,
    item_column: int,
    keep_users: set[str],
    stats: dict[str, object],
) -> None:
    users = stats["users"]
    items = stats["items"]

    with input_path.open("r", newline="") as source:
        reader = csv.reader(source, delimiter=sep)
        for row in reader:
            if row[user_column] not in keep_users:
                continue

            output_writer.writerow(row)
            stats["rows"] += 1
            users.add(row[user_column])
            items.add(row[item_column])


def materialize_splits(
    output_dir: Path,
    *,
    sep: str,
    write_header: bool,
    header_row: list[str] | None,
    user_column: int,
    item_column: int,
    train_users_pre_validation: set[str],
    train_users_between: set[str],
    validation_users: set[str],
    test_users: set[str],
    pre_validation_path: Path,
    between_path: Path,
    post_test_path: Path,
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_defs = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
    }
    stats = {
        name: {"rows": 0, "users": set(), "items": set()}
        for name in split_defs
    }

    with split_defs["train"].open("w", newline="") as train_file, split_defs[
        "validation"
    ].open("w", newline="") as validation_file, split_defs["test"].open(
        "w", newline=""
    ) as test_file:
        train_writer = csv.writer(train_file, delimiter=sep)
        validation_writer = csv.writer(validation_file, delimiter=sep)
        test_writer = csv.writer(test_file, delimiter=sep)

        if write_header and header_row is not None:
            train_writer.writerow(header_row)
            validation_writer.writerow(header_row)
            test_writer.writerow(header_row)

        append_filtered_rows(
            pre_validation_path,
            train_writer,
            sep=sep,
            user_column=user_column,
            item_column=item_column,
            keep_users=train_users_pre_validation,
            stats=stats["train"],
        )
        append_filtered_rows(
            between_path,
            train_writer,
            sep=sep,
            user_column=user_column,
            item_column=item_column,
            keep_users=train_users_between,
            stats=stats["train"],
        )
        append_filtered_rows(
            between_path,
            validation_writer,
            sep=sep,
            user_column=user_column,
            item_column=item_column,
            keep_users=validation_users,
            stats=stats["validation"],
        )
        append_filtered_rows(
            post_test_path,
            test_writer,
            sep=sep,
            user_column=user_column,
            item_column=item_column,
            keep_users=test_users,
            stats=stats["test"],
        )

    summary = {}
    for split_name, split_stats in stats.items():
        summary[split_name] = {
            "rows": split_stats["rows"],
            "users": len(split_stats["users"]),
            "items": len(split_stats["items"]),
        }
    return summary


def main() -> None:
    args = parse_args()
    validate_fraction("test_fraction", args.test_fraction)
    validate_fraction("validation_fraction", args.validation_fraction)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    header_row = None
    if args.write_header:
        if args.has_header:
            with dataset_path.open("r", newline="") as source:
                reader = csv.reader(source, delimiter=args.sep)
                header_row = next(reader, None)
        else:
            header_row = ["user_id", "item_id", "rating", "timestamp"]

    timestamps = load_sorted_timestamps(
        dataset_path,
        sep=args.sep,
        timestamp_column=args.timestamp_column,
        has_header=args.has_header,
    )
    test_timestamp, validation_timestamp = compute_cutoffs(
        timestamps,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )

    with tempfile.TemporaryDirectory(prefix="legacy_split_") as tempdir_name:
        tempdir = Path(tempdir_name)
        scan_result = scan_dataset(
            dataset_path,
            sep=args.sep,
            has_header=args.has_header,
            user_column=args.user_column,
            rating_column=args.rating_column,
            timestamp_column=args.timestamp_column,
            validation_timestamp=validation_timestamp,
            test_timestamp=test_timestamp,
            workdir=tempdir,
        )

        positive_before_validation = scan_result["positive_before_validation"]
        users_with_between = scan_result["users_with_between"]
        positive_after_test = scan_result["positive_after_test"]

        eligible_train_users = {
            user_id
            for user_id, positive_count in positive_before_validation.items()
            if positive_count > 1
        }
        legacy_train_users = eligible_train_users.intersection(positive_after_test)
        test_users = sample_users(
            legacy_train_users,
            args.max_test_users,
            args.random_seed,
        )
        validation_candidate_users = legacy_train_users.intersection(users_with_between)
        validation_users, validation_sampling_backend = sample_validation_users(
            validation_candidate_users,
            args.max_validation_users,
            args.random_seed,
        )
        train_users_pre_validation = legacy_train_users
        train_users_between = legacy_train_users.difference(validation_users)

        if args.stats_only:
            split_summary = materialize_splits(
                tempdir / "stats_only_output",
                sep=args.sep,
                write_header=args.write_header,
                header_row=header_row,
                user_column=args.user_column,
                item_column=args.item_column,
                train_users_pre_validation=train_users_pre_validation,
                train_users_between=train_users_between,
                validation_users=validation_users,
                test_users=test_users,
                pre_validation_path=scan_result["pre_validation_path"],
                between_path=scan_result["between_path"],
                post_test_path=scan_result["post_test_path"],
            )
            shutil.rmtree(tempdir / "stats_only_output")
        else:
            split_summary = materialize_splits(
                output_dir,
                sep=args.sep,
                write_header=args.write_header,
                header_row=header_row,
                user_column=args.user_column,
                item_column=args.item_column,
                train_users_pre_validation=train_users_pre_validation,
                train_users_between=train_users_between,
                validation_users=validation_users,
                test_users=test_users,
                pre_validation_path=scan_result["pre_validation_path"],
                between_path=scan_result["between_path"],
                post_test_path=scan_result["post_test_path"],
            )

        metadata = {
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "test_timestamp": test_timestamp,
            "validation_timestamp": validation_timestamp,
            "random_seed": args.random_seed,
            "max_test_users": args.max_test_users,
            "max_validation_users": args.max_validation_users,
            "legacy_train_users": len(legacy_train_users),
            "validation_candidate_users": len(validation_candidate_users),
            "validation_users_sampled": len(validation_users),
            "validation_sampling_backend": validation_sampling_backend,
            "test_users": len(test_users),
            "split_summary": split_summary,
        }

        print(json.dumps(metadata, indent=2, sort_keys=True))

        if not args.stats_only:
            metadata_path = output_dir / "split_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
