"""Frequency-domain Wiener deconvolution with a Gaussian blur-kernel baseline."""

from __future__ import annotations

import numpy as np


class WienerFilter:
    """Apply a regularized Wiener deconvolution to an RGB image in [0, 1]."""

    def __init__(
        self,
        kernel_size: int = 15,
        sigma: float = 2.5,
        noise_to_signal: float = 0.01,
    ) -> None:
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if not np.isfinite(noise_to_signal) or noise_to_signal < 0:
            raise ValueError("noise_to_signal must be non-negative and finite")
        self.kernel_size = kernel_size
        self.sigma = float(sigma)
        self.noise_to_signal = float(noise_to_signal)

    def _kernel(self) -> np.ndarray:
        radius = self.kernel_size // 2
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(coordinates, coordinates)
        kernel = np.exp(-(xx * xx + yy * yy) / (2 * self.sigma**2))
        return (kernel / kernel.sum()).astype(np.float32)

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore an HWC RGB float image and return an HWC float image in [0, 1]."""
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HWC RGB array with 3 channels")
        if min(image.shape[:2]) < 2:
            raise ValueError("image height and width must both be at least 2")
        if not np.isfinite(image).all():
            raise ValueError("image must contain only finite pixel values")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("image values must be in [0, 1]")

        radius = self.kernel_size // 2
        padded = np.pad(
            image,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )
        kernel = np.zeros(padded.shape[:2], dtype=np.float32)
        small_kernel = self._kernel()
        kernel[: self.kernel_size, : self.kernel_size] = small_kernel
        kernel = np.roll(kernel, shift=(-radius, -radius), axis=(0, 1))
        transfer = np.fft.fft2(kernel)[..., np.newaxis]
        spectrum = np.fft.fft2(padded, axes=(0, 1))
        inverse = np.conj(transfer) / (
            np.abs(transfer) ** 2 + self.noise_to_signal
        )
        restored = np.fft.ifft2(inverse * spectrum, axes=(0, 1)).real
        height, width = image.shape[:2]
        restored = restored[radius : radius + height, radius : radius + width]
        return np.clip(restored, 0, 1).astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.restore(image)
