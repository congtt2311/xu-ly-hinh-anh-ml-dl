
import numpy as np
import torch
from scipy.signal import wiener


def apply_wiener(image, size=5):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    result = np.zeros_like(image)

    if size % 2 == 0:
        size += 1

    for channel in range(3):

        filtered = wiener(
            image[:, :, channel],
            mysize=(size, size)
        )

        result[:, :, channel] = np.nan_to_num(
            filtered,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

    return np.clip(result, 0.0, 1.0)


def batch_wiener(images, size=5):

    device = images.device
    dtype = images.dtype

    images_numpy = (
        images.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    results = []

    for image in images_numpy:

        filtered = apply_wiener(
            image,
            size
        )

        results.append(filtered)

    results = np.stack(results)

    return (
        torch.from_numpy(results)
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=dtype)
    )
