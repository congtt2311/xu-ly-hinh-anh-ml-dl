"""Explore the prepared Unsplash image super-resolution dataset."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "eda"
SPLITS = ("train", "valid", "test")
SEED = 42


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path.relative_to(ROOT)}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected columns in {path.relative_to(ROOT)}: {reader.fieldnames}"
            )
        return list(reader)


def safe_processed_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid processed path in manifest: {relative_path}")
    path = PROCESSED / relative
    if not path.is_file():
        raise FileNotFoundError(f"Processed image not found: {relative_path}")
    return path


def image_metadata(path: Path) -> tuple[str, int, int, int]:
    with Image.open(path) as image:
        image_format = image.format or "unknown"
        width, height = image.size
    return image_format, width, height, path.stat().st_size


def load_records() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_pairs = read_csv(RAW / "image_data.csv", ["low_res", "high_res"])
    manifest = read_csv(
        PROCESSED / "manifest.csv",
        ["split", "LR", "HR", "source_LR", "source_HR"],
    )
    source_mapping = {
        row["low_res"]: row["high_res"]
        for row in source_pairs
        if row["low_res"] and row["high_res"]
    }

    if len(source_mapping) != len(source_pairs):
        raise ValueError("Source CSV contains empty or duplicate LR filenames")

    source_hr_splits: dict[str, set[str]] = defaultdict(set)
    records: list[dict[str, Any]] = []
    seen_lr: set[str] = set()

    for row_number, row in enumerate(manifest, start=2):
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown split on manifest row {row_number}: {split}")
        if row["source_LR"] in seen_lr:
            raise ValueError(
                f"Duplicate source LR filename on manifest row {row_number}: "
                f"{row['source_LR']}"
            )
        seen_lr.add(row["source_LR"])

        expected_hr = source_mapping.get(row["source_LR"])
        if expected_hr is None:
            raise ValueError(
                f"Source LR file is absent from image_data.csv: {row['source_LR']}"
            )
        if expected_hr != row["source_HR"]:
            raise ValueError(
                f"Source pair mismatch on manifest row {row_number}: "
                f"{row['source_LR']} maps to {expected_hr}, "
                f"not {row['source_HR']}"
            )

        lr_path = safe_processed_path(row["LR"])
        hr_path = safe_processed_path(row["HR"])
        if lr_path.name != hr_path.name:
            raise ValueError(
                f"Processed LR/HR names do not match on manifest row {row_number}"
            )

        lr_format, lr_width, lr_height, lr_bytes = image_metadata(lr_path)
        hr_format, hr_width, hr_height, hr_bytes = image_metadata(hr_path)
        suffix = Path(row["source_LR"]).stem.rsplit("_", maxsplit=1)
        variant = suffix[-1] if len(suffix) == 2 and suffix[-1].isdigit() else "unknown"
        source_hr_splits[row["source_HR"]].add(split)

        records.append(
            {
                **row,
                "variant": variant,
                "LR_format": lr_format,
                "HR_format": hr_format,
                "LR_width": lr_width,
                "LR_height": lr_height,
                "HR_width": hr_width,
                "HR_height": hr_height,
                "LR_bytes": lr_bytes,
                "HR_bytes": hr_bytes,
                "same_dimensions": lr_width == hr_width and lr_height == hr_height,
            }
        )

    if len(records) != len(source_pairs):
        raise ValueError(
            f"CSV/manifest pair count mismatch: {len(source_pairs)} source pairs, "
            f"{len(records)} manifest rows"
        )

    leaked_groups = [
        hr_name for hr_name, splits in source_hr_splits.items() if len(splits) > 1
    ]
    if leaked_groups:
        raise ValueError(
            f"{len(leaked_groups)} source HR images occur in multiple splits"
        )
    return records, source_pairs


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def image_quality_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_means = pixels.mean(axis=(0, 1))
    luminance = (
        0.2126 * pixels[:, :, 0]
        + 0.7152 * pixels[:, :, 1]
        + 0.0722 * pixels[:, :, 2]
    )
    return {
        "red_mean": float(channel_means[0]),
        "green_mean": float(channel_means[1]),
        "blue_mean": float(channel_means[2]),
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
    }


def sample_pairs(
    records: list[dict[str, Any]], samples_per_split: int
) -> list[dict[str, Any]]:
    randomizer = random.Random(SEED)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)

    samples = []
    for split in SPLITS:
        candidates = by_split[split]
        samples.extend(
            randomizer.sample(candidates, min(samples_per_split, len(candidates)))
        )

    result = []
    for record in samples:
        lr_path = PROCESSED / record["LR"]
        hr_path = PROCESSED / record["HR"]
        with Image.open(lr_path) as lr_image, Image.open(hr_path) as hr_image:
            lr = np.asarray(lr_image.convert("RGB"), dtype=np.float32)
            hr = np.asarray(hr_image.convert("RGB"), dtype=np.float32)

        if lr.shape != hr.shape:
            mse = float("nan")
            psnr = float("nan")
        else:
            mse = float(np.mean((lr - hr) ** 2))
            psnr = math.inf if mse == 0 else 10 * math.log10((255**2) / mse)

        lr_stats = image_quality_stats(lr_path)
        hr_stats = image_quality_stats(hr_path)
        result.append(
            {
                "split": record["split"],
                "LR": record["LR"],
                "HR": record["HR"],
                "variant": record["variant"],
                "MSE": mse,
                "PSNR_dB": psnr,
                "LR_luminance_mean": lr_stats["luminance_mean"],
                "HR_luminance_mean": hr_stats["luminance_mean"],
                "LR_luminance_std": lr_stats["luminance_std"],
                "HR_luminance_std": hr_stats["luminance_std"],
            }
        )
    return result


def plot_summaries(records: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(record["split"] for record in records)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        list(SPLITS),
        [split_counts[split] for split in SPLITS],
        color=["#4C78A8", "#F2CF5B", "#E45756"],
    )
    axis.set(title="Image pairs by split", xlabel="Split", ylabel="Pair count")
    axis.bar_label(axis.containers[0])
    fig.tight_layout()
    fig.savefig(OUTPUT / "pairs_by_split.png", dpi=150)
    plt.close(fig)

    variants = sorted({record["variant"] for record in records})
    fig, axis = plt.subplots(figsize=(7, 4))
    data = [
        [
            record["LR_bytes"] / 1024
            for record in records
            if record["variant"] == variant
        ]
        for variant in variants
    ]
    axis.boxplot(data, tick_labels=variants, showfliers=False)
    axis.set(
        title="Low-resolution file size by filename variant",
        xlabel="LR filename suffix",
        ylabel="File size (KiB)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "lr_file_size_by_variant.png", dpi=150)
    plt.close(fig)


def plot_sample_pairs(
    records: list[dict[str, Any]], samples_per_split: int
) -> list[dict[str, Any]]:
    randomizer = random.Random(SEED)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    samples = [
        record
        for split in SPLITS
        for record in randomizer.sample(
            by_split[split], min(samples_per_split, len(by_split[split]))
        )
    ]

    if not samples:
        return []

    fig, axes = plt.subplots(
        len(samples), 2, figsize=(10, max(3, 3 * len(samples))), squeeze=False
    )
    for row_index, record in enumerate(samples):
        for column_index, (label, relative_path) in enumerate(
            (("LR", record["LR"]), ("HR", record["HR"]))
        ):
            with Image.open(PROCESSED / relative_path) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].set_title(
                f"{record['split']} {label}: {Path(relative_path).name}"
            )
            axes[row_index, column_index].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "sample_pairs.png", dpi=150)
    plt.close(fig)
    return samples


def analyze(samples_per_split: int) -> None:
    records, source_pairs = load_records()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pair_columns = [
        "split",
        "source_LR",
        "source_HR",
        "variant",
        "LR",
        "HR",
        "LR_format",
        "HR_format",
        "LR_width",
        "LR_height",
        "HR_width",
        "HR_height",
        "LR_bytes",
        "HR_bytes",
        "same_dimensions",
    ]
    write_csv(OUTPUT / "pair_metadata.csv", records, pair_columns)

    split_counts = Counter(record["split"] for record in records)
    summary = []
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        summary.append(
            {
                "split": split,
                "pairs": len(split_records),
                "unique_HR_images": len(
                    {record["source_HR"] for record in split_records}
                ),
                "equal_size_pairs": sum(
                    record["same_dimensions"] for record in split_records
                ),
                "mean_LR_size_KiB": round(
                    float(np.mean([record["LR_bytes"] for record in split_records]))
                    / 1024,
                    2,
                )
                if split_records
                else 0,
                "mean_HR_size_KiB": round(
                    float(np.mean([record["HR_bytes"] for record in split_records]))
                    / 1024,
                    2,
                )
                if split_records
                else 0,
            }
        )
    write_csv(
        OUTPUT / "dataset_summary.csv",
        summary,
        [
            "split",
            "pairs",
            "unique_HR_images",
            "equal_size_pairs",
            "mean_LR_size_KiB",
            "mean_HR_size_KiB",
        ],
    )

    samples = sample_pairs(records, samples_per_split)
    write_csv(
        OUTPUT / "sample_image_statistics.csv",
        samples,
        [
            "split",
            "LR",
            "HR",
            "variant",
            "MSE",
            "PSNR_dB",
            "LR_luminance_mean",
            "HR_luminance_mean",
            "LR_luminance_std",
            "HR_luminance_std",
        ],
    )
    plot_summaries(records)
    plot_sample_pairs(records, samples_per_split)

    variants = Counter(record["variant"] for record in records)
    different_dimensions = sum(
        not record["same_dimensions"] for record in records
    )
    report_lines = [
        "Image Super Resolution EDA",
        "==========================",
        f"Source metadata pairs: {len(source_pairs)}",
        f"Validated processed pairs: {len(records)}",
        "Missing or mismatched pairs: 0",
        f"Pairs with different LR/HR dimensions: {different_dimensions}",
        f"LR filename variants: {dict(sorted(variants.items()))}",
        "",
        "Pairs and unique HR source images by split:",
    ]
    report_lines.extend(
        f"- {item['split']}: {item['pairs']} pairs, "
        f"{item['unique_HR_images']} unique HR images"
        for item in summary
    )
    report_lines.extend(
        [
            "",
            "Sample MSE/PSNR values are descriptive comparisons of the supplied "
            "LR/HR files; they are not model evaluation scores.",
            f"Artifacts: {OUTPUT.relative_to(ROOT)}",
        ]
    )
    (OUTPUT / "eda_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(report_lines))
    print("Generated: dataset_summary.csv, pair_metadata.csv,")
    print("           sample_image_statistics.csv, and PNG plots.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the prepared super-resolution image pairs."
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=2,
        help="Number of LR/HR pairs to inspect and plot from each split (default: 2).",
    )
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be at least 1")
    analyze(args.samples_per_split)


if __name__ == "__main__":
    main()