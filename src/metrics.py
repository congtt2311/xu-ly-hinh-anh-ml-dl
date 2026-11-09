
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_psnr(prediction, target):
    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    return peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0
    )


def calculate_ssim(prediction, target):
    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    min_size = min(prediction.shape[0], prediction.shape[1])
    win_size = min(7, min_size)

    if win_size % 2 == 0:
        win_size -= 1

    return structural_similarity(
        target,
        prediction,
        data_range=1.0,
        channel_axis=2,
        win_size=win_size
    )


def evaluate_metrics(prediction, target):
    psnr_value = calculate_psnr(prediction, target)
    ssim_value = calculate_ssim(prediction, target)

    return {
        "PSNR": psnr_value,
        "SSIM": ssim_value
    }
