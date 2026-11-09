"""Create a sequence-safe manifest for the GoPro deblurring dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "raw" / "GOPRO_Large"
MANIFEST = ROOT / "data" / "processed" / "gopro_manifest.csv"
SEED = 42
SPLITS = ("train", "valid", "test")
FIELDS = ("split", "sequence", "blur", "sharp", "blur_gamma")


def image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def collect_sequences(split: str) -> dict[str, list[dict[str, str]]]:
    split_dir = DATASET / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    sequences: dict[str, list[dict[str, str]]] = {}
    for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        blur_dir = sequence_dir / "blur"
        sharp_dir = sequence_dir / "sharp"
        blur_images = image_files(blur_dir)
        sharp_images = image_files(sharp_dir)
        if not blur_images:
            raise ValueError(f"No blur images found in {blur_dir}")
        if blur_images.keys() != sharp_images.keys():
            missing_sharp = sorted(blur_images.keys() - sharp_images.keys())
            missing_blur = sorted(sharp_images.keys() - blur_images.keys())
            raise ValueError(
                f"Unmatched blur/sharp images in {sequence_dir.name}: "
                f"missing sharp={missing_sharp[:5]}, missing blur={missing_blur[:5]}"
            )

        gamma_dir = sequence_dir / "blur_gamma"
        gamma_images = image_files(gamma_dir) if gamma_dir.is_dir() else {}
        if gamma_images and gamma_images.keys() != blur_images.keys():
            raise ValueError(f"Unmatched blur_gamma images in {sequence_dir.name}")

        sequences[sequence_dir.name] = [
            {
                "sequence": sequence_dir.name,
                "blur": blur_images[name].relative_to(ROOT).as_posix(),
                "sharp": sharp_images[name].relative_to(ROOT).as_posix(),
                "blur_gamma": (
                    gamma_images[name].relative_to(ROOT).as_posix()
                    if gamma_images
                    else ""
                ),
            }
            for name in sorted(blur_images)
        ]
    if not sequences:
        raise ValueError(f"No sequence folders found in {split_dir}")
    return sequences


def create_manifest(validation_sequences: int = 2) -> list[dict[str, str]]:
    train_sequences = collect_sequences("train")
    test_sequences = collect_sequences("test")
    if validation_sequences < 1 or validation_sequences >= len(train_sequences):
        raise ValueError(
            "validation_sequences must be between 1 and the number of training "
            "sequences minus one"
        )

    validation_names = set(
        random.Random(SEED).sample(
            sorted(train_sequences), k=validation_sequences
        )
    )
    rows = []
    for sequence, pairs in sorted(train_sequences.items()):
        split = "valid" if sequence in validation_names else "train"
        rows.extend({"split": split, **pair} for pair in pairs)
    for sequence, pairs in sorted(test_sequences.items()):
        rows.extend({"split": "test", **pair} for pair in pairs)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a GoPro blur/sharp manifest with sequence-level splits."
    )
    parser.add_argument(
        "--validation-sequences",
        type=int,
        default=2,
        help="Number of training sequences held out for validation (default: 2).",
    )
    args = parser.parse_args()

    rows = create_manifest(args.validation_sequences)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"Validation sequences: {args.validation_sequences}")
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        sequences = {row["sequence"] for row in split_rows}
        print(f"{split}: {len(split_rows)} pairs across {len(sequences)} sequences")


if __name__ == "__main__":
    main()
