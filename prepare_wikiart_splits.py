#!/usr/bin/env python3
"""Inspect the downloaded WikiArt subset and write a reproducible stratified split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


UNKNOWN_ARTIST = "Unknown Artist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/wikiart_style5"))
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inspect_and_read(data_dir: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    metadata_path = data_dir / "metadata.csv"
    subset_info_path = data_dir / "subset_info.json"
    if not metadata_path.is_file() or not subset_info_path.is_file():
        raise FileNotFoundError(f"Expected metadata.csv and subset_info.json under {data_dir}")

    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8", newline="")))
    subset_info = json.loads(subset_info_path.read_text(encoding="utf-8"))
    classes = sorted({row["label"] for row in rows})
    class_to_idx = {name: index for index, name in enumerate(classes)}
    errors: dict[str, list[str]] = {
        "duplicate_metadata_paths": [],
        "missing_images": [],
        "unlisted_images": [],
        "label_or_index_mismatches": [],
        "corrupt_images": [],
        "wrong_dimensions": [],
        "wrong_modes": [],
        "wrong_formats": [],
    }

    relative_paths = [row["path"] for row in rows]
    path_counts = Counter(relative_paths)
    errors["duplicate_metadata_paths"] = sorted(path for path, count in path_counts.items() if count > 1)
    listed_paths = set(relative_paths)
    disk_paths = {
        path.relative_to(data_dir).as_posix()
        for path in data_dir.glob("*/*.jpg")
        if path.is_file()
    }
    errors["missing_images"] = sorted(listed_paths - disk_paths)
    errors["unlisted_images"] = sorted(disk_paths - listed_paths)

    class_counts = Counter(row["label"] for row in rows)
    format_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    content_hashes: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        rel_path = Path(row["path"])
        expected_idx = class_to_idx.get(row["label"])
        if (
            rel_path.is_absolute()
            or ".." in rel_path.parts
            or row["label"] not in class_to_idx
            or row["label_idx"] != str(expected_idx)
            or rel_path.parts[0] != row["label"]
        ):
            errors["label_or_index_mismatches"].append(row["path"])

        image_path = data_dir / rel_path
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                format_name = image.format or "unknown"
                size = image.size
                mode = image.mode
            format_counts[format_name] += 1
            dimension_counts[f"{size[0]}x{size[1]}"] += 1
            mode_counts[mode] += 1
            if size != (128, 128):
                errors["wrong_dimensions"].append(row["path"])
            if mode != "RGB":
                errors["wrong_modes"].append(row["path"])
            if format_name != "JPEG":
                errors["wrong_formats"].append(row["path"])
            content_hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()].append(row["path"])
        except Exception as error:  # Keep the offending path in the quality report.
            errors["corrupt_images"].append(f"{row['path']}: {type(error).__name__}: {error}")

    duplicate_content = [paths for paths in content_hashes.values() if len(paths) > 1]
    artist_counts = Counter(row.get("artist", UNKNOWN_ARTIST) for row in rows)
    report: dict[str, object] = {
        "dataset": {
            "repo": subset_info["repo"],
            "revision": subset_info["revision"],
            "label": subset_info["label"],
            "classes": classes,
            "class_to_idx": class_to_idx,
            "metadata_rows": len(rows),
            "image_files": len(disk_paths),
            "class_counts": dict(sorted(class_counts.items())),
        },
        "image_integrity": {
            "format_counts": dict(sorted(format_counts.items())),
            "dimension_counts": dict(sorted(dimension_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
            "exact_duplicate_groups": len(duplicate_content),
            "exact_duplicate_images": sum(len(paths) for paths in duplicate_content),
            "errors": errors,
        },
        "artist_coverage": {
            "unknown_artist_images": artist_counts.get(UNKNOWN_ARTIST, 0),
            "unknown_artist_fraction": artist_counts.get(UNKNOWN_ARTIST, 0) / max(1, len(rows)),
            "named_artist_count": len([artist for artist in artist_counts if artist != UNKNOWN_ARTIST]),
        },
    }

    class_counts_match = all(class_counts[name] == subset_info["per_class"] for name in classes)
    report["image_integrity"]["class_counts_match_subset_plan"] = class_counts_match
    error_count = sum(len(values) for values in errors.values())
    if error_count or not class_counts_match or len(disk_paths) != len(rows):
        raise ValueError(
            f"Dataset inspection found {error_count} file/label errors; "
            f"class_counts_match_subset_plan={class_counts_match}."
        )
    return rows, report


def stratified_split(rows: list[dict[str, str]], seed: int) -> list[dict[str, object]]:
    classes = sorted({row["label"] for row in rows})
    class_to_idx = {name: index for index, name in enumerate(classes)}
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)

    rng = random.Random(seed)
    manifest: list[dict[str, object]] = []
    for label in classes:
        members = sorted(by_class[label], key=lambda row: row["path"])
        rng.shuffle(members)
        n_train = int(len(members) * 0.70)
        n_validation = int(len(members) * 0.15)
        boundaries = (
            ("train", members[:n_train]),
            ("validation", members[n_train:n_train + n_validation]),
            ("test", members[n_train + n_validation:]),
        )
        for split_name, selected in boundaries:
            for row in selected:
                manifest.append({
                    "relative_path": row["path"],
                    "label": label,
                    "label_idx": class_to_idx[label],
                    "split": split_name,
                    "seed": seed,
                    "artist": row.get("artist", UNKNOWN_ARTIST),
                })
    return sorted(manifest, key=lambda row: (row["label_idx"], row["split"], row["relative_path"]))


def main() -> None:
    args = parse_args()
    data_dir = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, report = inspect_and_read(data_dir)
    manifest = stratified_split(rows, args.seed)
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        split_counts[split_name] = dict(sorted(Counter(
            str(row["label"]) for row in manifest if row["split"] == split_name
        ).items()))

    artist_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        artist = str(row["artist"])
        if artist != UNKNOWN_ARTIST:
            artist_splits[artist].add(str(row["split"]))
    report["split"] = {
        "seed": args.seed,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "row_count": len(manifest),
        "counts": {split: sum(counts.values()) for split, counts in split_counts.items()},
        "class_counts": split_counts,
        "named_artists_present_in_multiple_splits": sum(len(splits) > 1 for splits in artist_splits.values()),
        "note": "Stratified by class. Artist grouping is not applied; Unknown Artist is a catch-all label.",
    }

    stem = f"{data_dir.name}_seed{args.seed}"
    manifest_path = output_dir / f"{stem}.csv"
    report_path = output_dir / f"{stem}_inspection.json"
    fieldnames = ("relative_path", "label", "label_idx", "split", "seed", "artist")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Inspecté : {len(rows)} images, {len(report['dataset']['classes'])} classes.")
    print("Formats :", report["image_integrity"]["format_counts"])
    print("Dimensions :", report["image_integrity"]["dimension_counts"])
    print("Modes :", report["image_integrity"]["mode_counts"])
    print("Doublons exacts :", report["image_integrity"]["exact_duplicate_groups"])
    print("Découpage :", report["split"]["counts"])
    print("Par classe :", report["split"]["class_counts"])
    print("Manifeste :", manifest_path)
    print("Rapport :", report_path)


if __name__ == "__main__":
    main()
