#!/usr/bin/env python3
"""Build a Yandex 30k-item dataset matching the Last.fm legacy workflow."""

from __future__ import annotations

import argparse
import csv
import random
import tarfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a WarpRec-compatible CSV from the Yandex Music event dump, "
            "keeping only one item type/event pair and sampling 30k items with "
            "popularity-weighted sampling without replacement."
        )
    )
    parser.add_argument(
        "--archive-path",
        default="data/yandex-music-event-2019-02-16/data.tar.gz",
        help="Path to the Yandex tar.gz archive.",
    )
    parser.add_argument(
        "--archive-member",
        default="user_events",
        help="Archive member containing the event table. Default: %(default)s",
    )
    parser.add_argument(
        "--output-path",
        default="data/yandex_30000/yandex_30000.csv",
        help="Path to the generated CSV file.",
    )
    parser.add_argument(
        "--item-type",
        default="track",
        help="Filter user_events rows by itemType. Default: %(default)s",
    )
    parser.add_argument(
        "--event",
        default="play",
        help="Filter user_events rows by event. Default: %(default)s",
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


def weighted_sample_without_replacement(
    items_with_counts: list[tuple[int, int]], sample_size: int, seed: int
) -> set[int]:
    """Fallback for NumPy choice(replace=False, p=...) using Efraimidis-Spirakis."""
    rng = random.Random(seed)
    weighted_keys: list[tuple[float, int]] = []
    for item_id, weight in items_with_counts:
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        key = u ** (1.0 / weight)
        weighted_keys.append((key, item_id))
    weighted_keys.sort(reverse=True)
    return {item_id for _, item_id in weighted_keys[:sample_size]}


def sample_items(items_with_counts: list[tuple[int, int]], sample_size: int, seed: int) -> set[int]:
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
        return {int(item_id) for item_id in sampled.tolist()}
    except ModuleNotFoundError:
        return weighted_sample_without_replacement(items_with_counts, sample_size, seed)


def iter_matching_rows(
    archive_path: Path,
    *,
    archive_member: str,
    item_type: str,
    event: str,
):
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(archive_member)
        if extracted is None:
            raise FileNotFoundError(
                f"Archive member {archive_member!r} not found in {archive_path}."
            )

        with TextIOWrapper(extracted, encoding="utf-8", newline="") as text_stream:
            reader = csv.DictReader(text_stream)
            for row_number, row in enumerate(reader, start=2):
                try:
                    if row["itemType"] != item_type or row["event"] != event:
                        continue

                    yield (
                        int(row["userId"]),
                        int(row["itemId"]),
                        1.0,
                        int(row["unixtime"]),
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid row at line {row_number}: {row!r}") from exc


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive_path)
    output_path = Path(args.output_path)

    item_counts = Counter(item_id for _, item_id, _, _ in iter_matching_rows(
        archive_path,
        archive_member=args.archive_member,
        item_type=args.item_type,
        event=args.event,
    ))
    sampled_items = sample_items(item_counts.most_common(), args.num_items, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    users: set[int] = set()
    kept_items: set[int] = set()
    rows_written = 0

    with output_path.open("w", newline="") as output:
        writer = csv.writer(output)
        if args.write_header:
            writer.writerow(["user_id", "item_id", "rating", "timestamp"])

        for user_id, item_id, rating, timestamp in iter_matching_rows(
            archive_path,
            archive_member=args.archive_member,
            item_type=args.item_type,
            event=args.event,
        ):
            if item_id not in sampled_items:
                continue

            writer.writerow([user_id, item_id, rating, timestamp])
            users.add(user_id)
            kept_items.add(item_id)
            rows_written += 1

    print(
        {
            "archive_path": str(archive_path),
            "archive_member": args.archive_member,
            "event": args.event,
            "item_type": args.item_type,
            "output_path": str(output_path),
            "rows": rows_written,
            "users": len(users),
            "items": len(kept_items),
            "sampling_seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
