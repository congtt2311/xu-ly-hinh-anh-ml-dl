"""PyTorch dataset and dataloader for the GoPro deblurring manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "gopro_manifest.csv"
REQUIRED_FIELDS = {"split", "sequence", "blur", "sharp", "blur_gamma"}
VALID_SPLITS = {"train", "valid", "test"}


class GoProDeblurDataset(Dataset[dict[str, Any]]):
    """Load paired GoPro blur/sharp images, optionally taking aligned crops."""

    def __init__(
        self,
        split: str,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        crop_size: int | tuple[int, int] | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        if crop_size is not None:
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
            if len(crop_size) != 2 or min(crop_size) < 1:
                raise ValueError("crop_size must contain two positive dimensions")
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        else:
            self.crop_size = None

        self.split = split
        self.manifest_path = Path(manifest_path).resolve()
        self.rows: list[dict[str, str]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if (
                reader.fieldnames is None
                or not REQUIRED_FIELDS.issubset(reader.fieldnames)
            ):
                raise ValueError(
                    f"Unexpected GoPro manifest columns in {self.manifest_path}"
                )
            for row_number, row in enumerate(reader, start=2):
                if row["split"] != split:
                    continue
                if not row["blur"] or not row["sharp"]:
                    raise ValueError(f"Empty image path on manifest row {row_number}")
                blur_path = self._resolve_path(row["blur"], row_number)
                sharp_path = self._resolve_path(row["sharp"], row_number)
                if not blur_path.is_file() or not sharp_path.is_file():
                    raise FileNotFoundError(
                        f"Image pair missing on manifest row {row_number}: "
                        f"{blur_path}, {sharp_path}"
                    )
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No {split!r} pairs in {self.manifest_path}")

    def _resolve_path(self, relative_path: str, row_number: int) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Invalid path on manifest row {row_number}: {relative_path}"
            )
        return (ROOT / path).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        blur_path = self._resolve_path(row["blur"], index + 2)
        sharp_path = self._resolve_path(row["sharp"], index + 2)
        with Image.open(blur_path) as image:
            blur = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(sharp_path) as image:
            sharp = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if blur.shape != sharp.shape:
            raise ValueError(
                f"Blur/sharp dimensions differ for "
                f"{row['sequence']}/{blur_path.name}"
            )

        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            height, width = blur.shape[:2]
            if height < crop_height or width < crop_width:
                raise ValueError(
                    f"Crop {self.crop_size} exceeds image size {(height, width)} "
                    f"for {row['sequence']}/{blur_path.name}"
                )
            top = int(torch.randint(height - crop_height + 1, ()).item())
            left = int(torch.randint(width - crop_width + 1, ()).item())
            blur = blur[top : top + crop_height, left : left + crop_width]
            sharp = sharp[top : top + crop_height, left : left + crop_width]

        return {
            "input": torch.from_numpy(blur.transpose(2, 0, 1).copy()),
            "target": torch.from_numpy(sharp.transpose(2, 0, 1).copy()),
            "sequence": row["sequence"],
            "filename": blur_path.name,
        }


def build_gopro_dataloader(
    split: str,
    batch_size: int = 1,
    crop_size: int | tuple[int, int] | None = 256,
    num_workers: int = 0,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> DataLoader[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    dataset = GoProDeblurDataset(
        split=split,
        manifest_path=manifest_path,
        crop_size=crop_size if split == "train" else None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
