
from pathlib import Path
import random
import cv2
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def find_image_pairs(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {root}")

    pairs = []

    for blur_dir in root.rglob("blur"):
        sharp_dir = blur_dir.parent / "sharp"

        if not sharp_dir.is_dir():
            continue

        sharp_files = {
            file.stem: file
            for file in sharp_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        }

        for blur_file in blur_dir.rglob("*"):
            if not blur_file.is_file() or blur_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            sharp_file = sharp_files.get(blur_file.stem)

            if sharp_file is not None:
                pairs.append((blur_file, sharp_file))

    pairs.sort(key=lambda item: str(item[0]))

    if not pairs:
        raise RuntimeError("Khong tim thay cap anh Blur va Sharp!")

    return pairs


def read_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Khong doc duoc anh: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class GoProDataset(Dataset):
    def __init__(self, root, patch_size=256, training=True):
        self.pairs = find_image_pairs(root)
        self.patch_size = patch_size
        self.training = training
        print(f"Da tim thay {len(self.pairs)} cap anh")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        blur_path, sharp_path = self.pairs[index]

        blur = read_image(blur_path)
        sharp = read_image(sharp_path)

        if blur.shape != sharp.shape:
            sharp = cv2.resize(sharp, (blur.shape[1], blur.shape[0]))

        if self.patch_size:
            height, width = blur.shape[:2]
            size = min(self.patch_size, height, width)

            if self.training:
                top = random.randint(0, height - size)
                left = random.randint(0, width - size)
            else:
                top = (height - size) // 2
                left = (width - size) // 2

            blur = blur[top:top + size, left:left + size]
            sharp = sharp[top:top + size, left:left + size]

            if self.training:
                if random.random() < 0.5:
                    blur = cv2.flip(blur, 1)
                    sharp = cv2.flip(sharp, 1)

                if random.random() < 0.5:
                    blur = cv2.flip(blur, 0)
                    sharp = cv2.flip(sharp, 0)

        blur = torch.from_numpy(blur.copy()).permute(2, 0, 1).float() / 255.0
        sharp = torch.from_numpy(sharp.copy()).permute(2, 0, 1).float() / 255.0

        return blur, sharp
