
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.model import CNNWienerDeblur
from src.wiener import apply_wiener


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay model: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = CNNWienerDeblur(
        features=checkpoint.get("features", 64),
        num_blocks=checkpoint.get("num_blocks", 8)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    wiener_size = checkpoint.get("wiener_size", 5)

    return model, wiener_size


def restore_image(image_path, model, device, wiener_size=5):
    try:
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
    except (OSError, ValueError) as error:
        raise FileNotFoundError(
            f"Khong doc duoc anh: {image_path}"
        ) from error
    image = image.astype(np.float32) / 255.0

    wiener_image = apply_wiener(image, wiener_size)

    if model is None:
        return image, wiener_image, wiener_image

    blur_tensor = torch.from_numpy(
        image.copy()
    ).permute(2, 0, 1).unsqueeze(0).float().to(device)

    wiener_tensor = torch.from_numpy(
        wiener_image.copy()
    ).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        restored = model(blur_tensor, wiener_tensor)

    restored = restored.squeeze(0).cpu().permute(1, 2, 0).numpy()

    restored = np.clip(restored, 0.0, 1.0)

    return image, wiener_image, restored


def save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        Image.fromarray(
            np.clip(image * 255.0, 0, 255).astype(np.uint8),
            mode="RGB",
        ).save(output_path)
    except OSError as error:
        raise OSError(f"Khong the luu anh: {output_path}") from error
