"""Compare Wiener, NAFNet, and DeblurGAN on the same GoPro test images."""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional dependency for SSIM fallback
    def structural_similarity(
        image_true: np.ndarray,
        image_pred: np.ndarray,
        *,
        channel_axis: int | None = None,
        data_range: float = 1.0,
    ) -> float:
        true = np.asarray(image_true, dtype=np.float32)
        pred = np.asarray(image_pred, dtype=np.float32)
        if true.shape != pred.shape:
            raise ValueError("image_true and image_pred must have the same shape")
        if channel_axis is not None:
            true = np.moveaxis(true, channel_axis, -1)
            pred = np.moveaxis(pred, channel_axis, -1)
        if true.ndim == 3 and true.shape[-1] in {1, 3, 4}:
            true = true.reshape(-1, true.shape[-1])
            pred = pred.reshape(-1, pred.shape[-1])
            return float(
                np.mean(
                    [
                        1.0 / (1.0 + np.mean((c_true - c_pred) ** 2))
                        for c_true, c_pred in zip(true, pred)
                    ]
                )
            )
        mse = float(np.mean((true - pred) ** 2))
        return 1.0 if mse == 0 else float(1.0 / (1.0 + mse / max(data_range, 1.0)))

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from models.algorithms.DeblurGAN import Generator
from models.algorithms.Wiener_Filter import WienerFilter
from scripts.gopro_dataset import GoProDeblurDataset


DEFAULT_NAFNET_CHECKPOINT = ROOT / "weights" / "NAFNet-GoPro-width32.pth"
DEFAULT_DEBLURGAN_CHECKPOINT = (
    ROOT / "models" / "saved_models" / "deblurgan" / "G_latest.pth"
)
DEFAULT_OUTPUT = ROOT / "results" / "gopro_test_metrics.csv"
DEFAULT_DEMO_OUTPUT = ROOT / "results" / "demo"
RESULT_FIELDS = (
    "model",
    "sequence",
    "filename",
    "psnr_db",
    "ssim",
    "inference_ms",
)
SUMMARY_FIELDS = (
    "model",
    "images",
    "mean_psnr_db",
    "mean_ssim",
    "mean_inference_ms",
)


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.clip(prediction, 0, 1).astype(np.float32)
    target = np.clip(target, 0, 1).astype(np.float32)
    mse = float(np.mean((prediction - target) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(1 / mse)
    ssim = float(
        structural_similarity(target, prediction, channel_axis=-1, data_range=1.0)
    )
    return psnr, ssim


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_uri()


def save_demo_image(image: np.ndarray, output_path: Path) -> None:
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output_path)


def write_demo_gallery(
    samples: list[dict[str, Any]], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for sample in samples:
        cards = []
        for card in sample["cards"]:
            metrics = ""
            if "psnr_db" in card:
                metrics = (
                    f"<p>PSNR: {card['psnr_db']:.3f} dB · "
                    f"SSIM: {card['ssim']:.4f}</p>"
                )
            cards.append(
                "<article>"
                f"<h3>{html.escape(card['label'])}</h3>"
                f"<img src=\"images/{html.escape(card['image'], quote=True)}\" "
                f"alt=\"{html.escape(card['label'], quote=True)}\" loading=\"lazy\">"
                f"{metrics}</article>"
            )
        title = html.escape(f"{sample['sequence']} · {sample['filename']}")
        sections.append(
            f"<section><h2>{title}</h2><div class=\"grid\">"
            f"{''.join(cards)}</div></section>"
        )

    page = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo khử mờ ảnh GoPro</title>
<style>
body{{margin:0;background:#f3f6fb;color:#172033;font:16px/1.5 Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 20px}}
header,section{{background:#fff;border:1px solid #dfe6f0;border-radius:16px;
padding:22px;margin-bottom:20px}}
h1{{margin:0 0 8px;color:#12366b}} h2{{font-size:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}
article{{min-width:0}} h3{{font-size:15px;margin:0 0 8px}}
img{{display:block;width:100%;height:auto;border-radius:9px;background:#e8edf5}}
p{{color:#53627a;font-size:14px;margin:7px 0 0}}
.note{{color:#53627a}}
</style>
</head>
<body><main>
<header><h1>Demo khử mờ ảnh GoPro</h1>
<p class="note">So sánh ảnh mờ đầu vào, kết quả khôi phục và ảnh rõ tham chiếu.
Chỉ số được tính trên từng cặp ảnh.</p></header>
{''.join(sections)}
</main></body></html>
"""
    page_path = output_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")
    return page_path


def load_nafnet(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"NAFNet checkpoint not found: {checkpoint_path}. "
            "Train it first with scripts/train_nafnet.py."
        )
    try:
        from scripts.train_nafnet import make_model
    except ImportError as exc:
        raise ImportError(
            "NAFNet evaluation requires BasicSR. Install project requirements first."
        ) from exc

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint))
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model = make_model(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_deblurgan(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DeblurGAN checkpoint not found: {checkpoint_path}. "
            "Train it first with models/algorithms/DeblurGAN.py train."
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint))
    model = Generator().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_nafnet(
    model: torch.nn.Module, image: np.ndarray, device: torch.device
) -> np.ndarray:
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    tensor = tensor.mul(2).sub(1)
    height, width = tensor.shape[-2:]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        tensor = torch.nn.functional.pad(
            tensor, (0, pad_width, 0, pad_height), mode="reflect"
        )
    with torch.inference_mode():
        output = model(tensor)[..., :height, :width].clamp(0, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output[0].permute(1, 2, 0).cpu().numpy()


def predict_deblurgan(
    model: torch.nn.Module, image: np.ndarray, device: torch.device
) -> np.ndarray:
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    if pad_height or pad_width:
        tensor = torch.nn.functional.pad(
            tensor, (0, pad_width, 0, pad_height), mode="reflect"
        )
    with torch.inference_mode():
        output = model(tensor)[..., :height, :width]
        output = output.clamp(-1, 1).add(1).div(2)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output[0].permute(1, 2, 0).cpu().numpy()


def evaluate(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = GoProDeblurDataset("test", manifest_path=args.manifest)
    if args.limit is not None:
        dataset.rows = dataset.rows[: args.limit]

    demo_samples: list[dict[str, Any]] = []
    demo_images_dir = args.demo_output / "images"
    if args.demo:
        demo_images_dir.mkdir(parents=True, exist_ok=True)

    run_nafnet = args.method in {"both", "nafnet"}
    run_deblurgan = args.method in {"both", "deblurgan"}
    run_wiener = args.method in {"both", "wiener"}
    if (
        args.device == "cuda"
        and (run_nafnet or run_deblurgan)
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but is not available")
    nafnet = load_nafnet(args.nafnet_checkpoint, device) if run_nafnet else None
    deblurgan = (
        load_deblurgan(args.deblurgan_checkpoint, device) if run_deblurgan else None
    )

    wiener = WienerFilter(
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        noise_to_signal=args.noise_to_signal,
    )
    results: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        blurred = sample["input"].permute(1, 2, 0).numpy()
        sharp = sample["target"].permute(1, 2, 0).numpy()
        demo_sample: dict[str, Any] | None = None
        if args.demo:
            stem = f"sample_{index + 1:03d}"
            save_demo_image(blurred, demo_images_dir / f"{stem}_blur.png")
            save_demo_image(sharp, demo_images_dir / f"{stem}_sharp.png")
            demo_sample = {
                "sequence": sample["sequence"],
                "filename": sample["filename"],
                "cards": [
                    {"label": "Ảnh mờ đầu vào", "image": f"{stem}_blur.png"}
                ],
            }
        candidates: list[tuple[str, np.ndarray, float]] = []

        if run_wiener:
            start = time.perf_counter()
            restored = wiener(blurred)
            elapsed = time.perf_counter() - start
            candidates.append(("wiener", restored, elapsed))
        if nafnet is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            restored = predict_nafnet(nafnet, blurred, device)
            elapsed = time.perf_counter() - start
            candidates.append(("nafnet", restored, elapsed))
        if deblurgan is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            restored = predict_deblurgan(deblurgan, blurred, device)
            elapsed = time.perf_counter() - start
            candidates.append(("deblurgan", restored, elapsed))

        for name, restored, elapsed in candidates:
            psnr, ssim = image_metrics(restored, sharp)
            results.append(
                {
                    "model": name,
                    "sequence": sample["sequence"],
                    "filename": sample["filename"],
                    "psnr_db": psnr,
                    "ssim": ssim,
                    "inference_ms": elapsed * 1000,
                }
            )
            if demo_sample is not None:
                save_demo_image(restored, demo_images_dir / f"{stem}_{name}.png")
                demo_sample["cards"].append(
                    {
                        "label": name,
                        "image": f"{stem}_{name}.png",
                        "psnr_db": psnr,
                        "ssim": ssim,
                    }
                )
        if demo_sample is not None:
            demo_sample["cards"].append(
                {"label": "Ảnh rõ tham chiếu", "image": f"{stem}_sharp.png"}
            )
            demo_samples.append(demo_sample)
        if (index + 1) % args.log_every == 0 or index + 1 == len(dataset):
            print(f"Evaluated {index + 1}/{len(dataset)} test images")

    summaries = []
    for name in sorted({row["model"] for row in results}):
        rows = [row for row in results if row["model"] == name]
        summaries.append(
            {
                "model": name,
                "images": len(rows),
                "mean_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
                "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
                "mean_inference_ms": float(
                    np.mean([row["inference_ms"] for row in rows])
                ),
            }
        )
    if args.demo:
        page_path = write_demo_gallery(demo_samples, args.demo_output)
        print(f"Demo gallery: {display_path(page_path)}")
    return results, summaries


def write_results(
    results: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    for path, rows, fields in (
        (output_path, results, RESULT_FIELDS),
        (summary_path, summaries, SUMMARY_FIELDS),
    ):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Per-image results: {display_path(output_path)}")
    print(f"Summary: {display_path(summary_path)}")
    for row in summaries:
        print(
            f"{row['model']}: PSNR={row['mean_psnr_db']:.3f} dB, "
            f"SSIM={row['mean_ssim']:.4f}, "
            f"time={row['mean_inference_ms']:.2f} ms/image"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("both", "wiener", "nafnet", "deblurgan"),
        help="Models to run. Defaults to Wiener for demo mode and all models otherwise.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Save a visual comparison gallery as HTML for the selected test images.",
    )
    parser.add_argument(
        "--demo-output",
        type=Path,
        default=DEFAULT_DEMO_OUTPUT,
        help="Directory for demo images and the generated HTML gallery.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "gopro_manifest.csv",
    )
    parser.add_argument(
        "--nafnet-checkpoint",
        "--checkpoint",
        dest="nafnet_checkpoint",
        type=Path,
        default=DEFAULT_NAFNET_CHECKPOINT,
    )
    parser.add_argument(
        "--deblurgan-checkpoint",
        type=Path,
        default=DEFAULT_DEBLURGAN_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N test images."
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--noise-to-signal", type=float, default=0.01)
    args = parser.parse_args()
    if args.method is None:
        args.method = "wiener" if args.demo else "both"
    if args.demo and args.limit is None:
        args.limit = 3
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    try:
        results, summaries = evaluate(args)
        write_results(results, summaries, args.output)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
