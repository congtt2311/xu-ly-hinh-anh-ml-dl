import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
IMAGE_SOURCE = RAW / "dataset" / "Image Super Resolution - Unsplash"
CSV_PATH = RAW / "image_data.csv"
PROCESSED = ROOT / "data" / "processed"
SPLIT_SEED = 42


def load_pairs():
    pairs = []
    missing = []

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["low_res", "high_res"]:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

        seen_lr = set()
        for row_number, row in enumerate(reader, start=2):
            lr_name = row["low_res"]
            hr_name = row["high_res"]
            if not lr_name or not hr_name:
                raise ValueError(f"Empty filename on CSV row {row_number}")
            if Path(lr_name).name != lr_name or Path(hr_name).name != hr_name:
                raise ValueError(f"Invalid filename on CSV row {row_number}")
            if lr_name in seen_lr:
                raise ValueError(f"Duplicate LR filename on CSV row {row_number}: {lr_name}")
            seen_lr.add(lr_name)

            lr_path = IMAGE_SOURCE / "low res" / lr_name
            hr_path = IMAGE_SOURCE / "high res" / hr_name
            if not lr_path.is_file():
                missing.append(f"LR: {lr_path}")
            if not hr_path.is_file():
                missing.append(f"HR: {hr_path}")
            pairs.append((lr_name, hr_name, lr_path, hr_path))

    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(f"Missing files referenced by image_data.csv:\n{details}")
    return pairs


def split_by_high_res(pairs):
    pairs_by_hr = defaultdict(list)
    for pair in pairs:
        pairs_by_hr[pair[1]].append(pair)

    hr_names = sorted(pairs_by_hr)
    random.Random(SPLIT_SEED).shuffle(hr_names)
    train_end = round(len(hr_names) * 0.8)
    valid_end = train_end + round(len(hr_names) * 0.1)
    hr_splits = {
        "train": hr_names[:train_end],
        "valid": hr_names[train_end:valid_end],
        "test": hr_names[valid_end:],
    }

    return {
        split: [
            pair
            for hr_name in names
            for pair in pairs_by_hr[hr_name]
        ]
        for split, names in hr_splits.items()
    }


def prepare():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")
    if not (IMAGE_SOURCE / "low res").is_dir() or not (IMAGE_SOURCE / "high res").is_dir():
        raise FileNotFoundError(
            f"Source image folders not found under {IMAGE_SOURCE}. "
            "Restore the extracted high res and low res folders before preparing data."
        )

    existing = [path.name for path in PROCESSED.iterdir() if path.name != ".gitkeep"]
    if existing:
        raise FileExistsError(
            f"{PROCESSED} is not empty (excluding .gitkeep): {', '.join(existing)}"
        )

    pairs = load_pairs()
    splits = split_by_high_res(pairs)
    manifest_path = PROCESSED / "manifest.csv"

    for split, split_pairs in splits.items():
        lr_output = PROCESSED / split / "LR"
        hr_output = PROCESSED / split / "HR"
        lr_output.mkdir(parents=True)
        hr_output.mkdir(parents=True)

        for lr_name, _hr_name, lr_path, hr_path in split_pairs:
            shutil.copy2(lr_path, lr_output / lr_name)
            shutil.copy2(hr_path, hr_output / lr_name)

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "LR", "HR", "source_LR", "source_HR"])
        for split, split_pairs in splits.items():
            for lr_name, hr_name, _lr_path, _hr_path in split_pairs:
                writer.writerow(
                    [split, f"{split}/LR/{lr_name}", f"{split}/HR/{lr_name}", lr_name, hr_name]
                )

    unique_hr = len({hr_name for _, hr_name, _, _ in pairs})
    print(f"Valid pairs: {len(pairs)}")
    print(f"Unique HR source images: {unique_hr}")
    print("Missing LR files: 0")
    print("Missing HR files: 0")
    for split, split_pairs in splits.items():
        unique_split_hr = len({hr_name for _, hr_name, _, _ in split_pairs})
        print(
            f"{split}: {len(split_pairs)} pairs across "
            f"{unique_split_hr} HR source images"
        )
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    prepare()
