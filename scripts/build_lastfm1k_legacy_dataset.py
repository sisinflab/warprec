#!/usr/bin/env python3
"""Build a Last.fm 30k-item dataset matching the legacy loader semantics."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CSV dataset compatible with WarpRec while preserving the "
            "legacy Last.fm loader semantics: keep only rows with non-empty "
            "musicbrainz_track_id and sample 30k items with popularity-weighted "
            "sampling without replacement."
        )
    )
    parser.add_argument(
        "--input-path",
        default="data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv",
        help="Path to the raw Last.fm TSV file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/lastfm1k_30000/lastfm_legacy_mbids_30000.csv",
        help="Path to the generated CSV file.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=30000,
        help="Number of items to sample. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Sampling seed. Default: %(default)s",
    )
    parser.add_argument(
        "--write-header",
        action="store_true",
        help="Write a header row to the output CSV.",
    )
    return parser.parse_args()


def weighted_sample_without_replacement(items_with_counts: list[tuple[str, int]], sample_size: int, seed: int) -> set[str]:
    """Fallback for NumPy choice(replace=False, p=...) using Efraimidis-Spirakis."""
    rng = random.Random(seed)
    weighted_keys: list[tuple[float, str]] = []
    for item_id, weight in items_with_counts:
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        key = u ** (1.0 / weight)
        weighted_keys.append((key, item_id))
    weighted_keys.sort(reverse=True)
    return {item_id for _, item_id in weighted_keys[:sample_size]}


def sample_items(items_with_counts: list[tuple[str, int]], sample_size: int, seed: int) -> set[str]:
    try:
        import numpy as np  # type: ignore

        item_ids = [item_id for item_id, _ in items_with_counts]
        counts = [count for _, count in items_with_counts]
        total = sum(counts)
        probabilities = [count / total for count in counts]
        np.random.seed(seed)
        sampled = np.random.choice(
            a=item_ids,
            replace=False,
            p=probabilities,
            size=min(sample_size, len(item_ids)),
        )
        return set(sampled.tolist())
    except ModuleNotFoundError:
        return weighted_sample_without_replacement(items_with_counts, sample_size, seed)


def load_rows(input_path: Path) -> list[tuple[int, str, float, str]]:
    csv.field_size_limit(sys.maxsize)
    rows: list[tuple[int, str, float, str]] = []
    with input_path.open("r", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for row in reader:
            if len(row) != 6:
                continue
            user_id, timestamp, _, _, musicbrainz_track_id, _ = row
            if not musicbrainz_track_id:
                continue
            rows.append(
                (
                    int(user_id.replace("user_", "")),
                    musicbrainz_track_id,
                    1.0,
                    timestamp,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    rows = load_rows(input_path)
    item_counts = Counter(item_id for _, item_id, _, _ in rows).most_common()
    sampled_items = sample_items(item_counts, args.num_items, args.seed)
    filtered_rows = [row for row in rows if row[1] in sampled_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as output:
        writer = csv.writer(output)
        if args.write_header:
            writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for user_id, item_id, rating, timestamp in filtered_rows:
            writer.writerow([user_id, item_id, rating, timestamp])

    print(
        {
            "output_path": str(output_path),
            "rows": len(filtered_rows),
            "users": len({user_id for user_id, _, _, _ in filtered_rows}),
            "items": len({item_id for _, item_id, _, _ in filtered_rows}),
            "sampling_seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
