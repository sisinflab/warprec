#!/usr/bin/env python3
"""Fingerprint the legacy Last.fm 30k-item dataset construction pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.build_lastfm1k_legacy_dataset import load_rows, sample_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reproducible fingerprints for the legacy Last.fm dataset "
            "builder so two environments can be compared exactly."
        )
    )
    parser.add_argument(
        "--input-path",
        default="data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv",
        help="Path to the raw Last.fm TSV file.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=30000,
        help="Number of sampled items. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Sampling seed. Default: %(default)s",
    )
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Optional path where the fingerprint JSON will be written.",
    )
    parser.add_argument(
        "--write-item-list",
        type=str,
        default=None,
        help="Optional path where the sampled sorted item ids will be written.",
    )
    return parser.parse_args()


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def raw_file_stats(path: Path) -> dict[str, int]:
    csv.field_size_limit(sys.maxsize)
    rows_total = 0
    rows_non_empty_mbid = 0
    users = set()
    items = set()

    with path.open("r", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for row in reader:
            rows_total += 1
            if len(row) != 6:
                continue
            user_id, _timestamp, _artid, _artname, musicbrainz_track_id, _traname = row
            if not musicbrainz_track_id:
                continue
            rows_non_empty_mbid += 1
            users.add(int(user_id.replace("user_", "")))
            items.add(musicbrainz_track_id)

    return {
        "rows_total": rows_total,
        "rows_non_empty_mbid": rows_non_empty_mbid,
        "users": len(users),
        "items": len(items),
    }


def digest_sampled_items(sampled_items: set[str]) -> str:
    digest = hashlib.sha256()
    for item_id in sorted(sampled_items):
        digest.update(item_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def digest_filtered_rows(filtered_rows: list[tuple[int, str, float, str]]) -> str:
    digest = hashlib.sha256()
    for user_id, item_id, rating, timestamp in filtered_rows:
        digest.update(f"{user_id},{item_id},{rating},{timestamp}\n".encode("utf-8"))
    return digest.hexdigest()


def top_item_counts(rows: list[tuple[int, str, float, str]], limit: int = 10) -> list[list[str | int]]:
    counts = Counter(item_id for _user_id, item_id, _rating, _timestamp in rows)
    return [[item_id, count] for item_id, count in counts.most_common(limit)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)

    raw_stats = raw_file_stats(input_path)
    rows = load_rows(input_path)
    item_counts = Counter(item_id for _user_id, item_id, _rating, _timestamp in rows).most_common()
    sampled_items = sample_items(item_counts, args.num_items, args.seed)
    filtered_rows = [row for row in rows if row[1] in sampled_items]

    result = {
        "input_path": str(input_path),
        "raw_file_sha256": sha256_of_file(input_path),
        "raw_stats": raw_stats,
        "sampling_seed": args.seed,
        "num_items_requested": args.num_items,
        "sampled_item_count": len(sampled_items),
        "sampled_item_sha256": digest_sampled_items(sampled_items),
        "filtered_rows_sha256": digest_filtered_rows(filtered_rows),
        "filtered_stats": {
            "rows": len(filtered_rows),
            "users": len({user_id for user_id, _item_id, _rating, _timestamp in filtered_rows}),
            "items": len({item_id for _user_id, item_id, _rating, _timestamp in filtered_rows}),
        },
        "top_item_counts_after_mbid_filter": top_item_counts(rows),
        "top_item_counts_after_sampling": top_item_counts(filtered_rows),
    }

    if args.write_json:
        output_path = Path(args.write_json)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.write_item_list:
        item_list_path = Path(args.write_item_list)
        item_list_path.write_text("".join(f"{item_id}\n" for item_id in sorted(sampled_items)))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
