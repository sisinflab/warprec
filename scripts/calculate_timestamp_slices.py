#!/usr/bin/env python3
"""Compute timestamp_slicing cutoffs matching the provided GlobalTemporalSplit."""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute test/validation timestamp cutoffs for WarpRec-style "
            "timestamp_slicing using the same indexing logic as the provided "
            "GlobalTemporalSplit snippet."
        )
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="data/lastfm1k_30000/lastfm_no_header.csv",
        help="Path to the interaction file. Default: %(default)s",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Column separator used by the dataset. Default: %(default)r",
    )
    parser.add_argument(
        "--timestamp-column",
        type=int,
        default=3,
        help="Zero-based index of the timestamp column. Default: %(default)s",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Skip the first row as header.",
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
        help="Fraction reserved for validation on the pre-test partition. Default: %(default)s",
    )
    return parser.parse_args()


def validate_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in the open interval (0, 1), got {value}.")


def load_timestamps(
    dataset_path: Path,
    *,
    sep: str,
    timestamp_column: int,
    has_header: bool,
) -> list[int]:
    timestamps: list[int] = []

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

            try:
                timestamps.append(int(row[timestamp_column]))
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number} has a non-integer timestamp: {row[timestamp_column]!r}"
                ) from exc

    if not timestamps:
        raise ValueError("The dataset does not contain any timestamp values.")

    timestamps.sort()
    return timestamps


def compute_cutoffs(
    timestamps: list[int],
    *,
    test_fraction: float,
    validation_fraction: float,
) -> dict[str, int]:
    total = len(timestamps)
    test_index = int(total * (1.0 - test_fraction))
    if test_index >= total:
        raise ValueError(
            f"Test index {test_index} is out of range for {total} timestamps. "
            "Decrease test_fraction."
        )

    test_timestamp = timestamps[test_index]

    # Match the provided code exactly: the pre-test subset uses a strict < test_timestamp check.
    pre_test_count = bisect.bisect_left(timestamps, test_timestamp)
    if pre_test_count == 0:
        raise ValueError("No timestamps fall before the computed test border timestamp.")

    val_index = int(pre_test_count * (1.0 - validation_fraction))
    if val_index >= total:
        raise ValueError(
            f"Validation index {val_index} is out of range for {total} timestamps. "
            "Decrease validation_fraction."
        )

    # Match the provided snippet exactly: the index is computed on the pre-test length,
    # then applied to the globally sorted list.
    validation_timestamp = timestamps[val_index]

    result = {
        "total_rows": total,
        "test_index": test_index,
        "test_timestamp": test_timestamp,
        "pre_test_count": pre_test_count,
        "validation_index": val_index,
        "validation_timestamp": validation_timestamp,
    }

    if pre_test_count != test_index:
        result["duplicate_border_timestamps"] = test_index - pre_test_count

    return result


def main() -> None:
    args = parse_args()
    validate_fraction("test_fraction", args.test_fraction)
    validate_fraction("validation_fraction", args.validation_fraction)

    dataset_path = Path(args.dataset_path)
    timestamps = load_timestamps(
        dataset_path,
        sep=args.sep,
        timestamp_column=args.timestamp_column,
        has_header=args.has_header,
    )
    result = compute_cutoffs(
        timestamps,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )

    print(f"dataset_path: {dataset_path}")
    print(f"total_rows: {result['total_rows']}")
    print(f"test_index: {result['test_index']}")
    print(f"test_timestamp: {result['test_timestamp']}")
    print(f"pre_test_count (< test_timestamp): {result['pre_test_count']}")
    print(f"validation_index: {result['validation_index']}")
    print(f"validation_timestamp: {result['validation_timestamp']}")
    if "duplicate_border_timestamps" in result:
        print(
            "warning: duplicate timestamps were found at the test border; "
            "the validation timestamp still follows the exact GlobalTemporalSplit indexing."
        )

    print()
    print("splitter:")
    print("  test_splitting:")
    print("    strategy: timestamp_slicing")
    print(
        f"    timestamp: {result['test_timestamp']}   "
        f"# Computed with test_fraction={args.test_fraction}"
    )
    print("  validation_splitting:")
    print("    strategy: timestamp_slicing")
    print(
        f"    timestamp: {result['validation_timestamp']}   "
        f"# Computed with validation_fraction={args.validation_fraction}"
    )


if __name__ == "__main__":
    main()
