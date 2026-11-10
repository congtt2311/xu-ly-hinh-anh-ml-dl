
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CHECKPOINT_PATH,
    MANIFEST_PATH,
    NUM_WORKERS,
    OUTPUT_PATH,
    WIENER_SIZE,
)
from scripts.gopro_dataset import GoProDeblurDataset
from src.model import CNNWienerDeblur
from src.wiener import batch_wiener
from src.metrics import evaluate_metrics
from src.report import generate_multipage_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Center-crop size; use 0 to evaluate full-resolution images.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.patch_size < 0:
        parser.error("--patch-size cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Khong tim thay model: {checkpoint_file}. Hay chay train.py truoc."
        )

    checkpoint = torch.load(
        checkpoint_file,
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

    wiener_size = checkpoint.get("wiener_size", WIENER_SIZE)

    dataset = GoProDeblurDataset(
        "test",
        manifest_path=args.manifest,
        crop_size=args.patch_size or None,
    )

    if args.limit > 0:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(args.limit, len(dataset))),
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    results = {
        "Blur": [],
        "Wiener": [],
        "CNN + Wiener": []
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            blur = batch["input"]
            sharp = batch["target"]
            blur = blur.to(device)
            sharp = sharp.to(device)

            wiener = batch_wiener(
                blur,
                size=wiener_size
            )

            prediction = model(blur, wiener)

            blur_np = blur.squeeze(0).cpu().permute(1, 2, 0).numpy()
            sharp_np = sharp.squeeze(0).cpu().permute(1, 2, 0).numpy()
            wiener_np = wiener.squeeze(0).cpu().permute(1, 2, 0).numpy()
            prediction_np = prediction.squeeze(0).cpu().permute(1, 2, 0).numpy()

            results["Blur"].append(
                evaluate_metrics(blur_np, sharp_np)
            )

            results["Wiener"].append(
                evaluate_metrics(wiener_np, sharp_np)
            )

            results["CNN + Wiener"].append(
                evaluate_metrics(prediction_np, sharp_np)
            )

    print("\n========== EVALUATION RESULTS ==========")

    final_results = []

    for method, values in results.items():
        psnr = float(np.mean([item["PSNR"] for item in values]))
        ssim = float(np.mean([item["SSIM"] for item in values]))

        print(f"\n{method}")
        print(f"Average PSNR: {psnr:.4f} dB")
        print(f"Average SSIM: {ssim:.4f}")

        final_results.append({
            "Method": method,
            "PSNR": psnr,
            "SSIM": ssim
        })

    output_dir = OUTPUT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "evaluation_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Method", "Images", "PSNR", "SSIM"]
        )
        writer.writeheader()
        for row in final_results:
            row["Images"] = len(results[row["Method"]])
        writer.writerows(final_results)

    report_path = generate_multipage_report(
        output_path,
        manifest_path=args.manifest,
    )

    print(f"\nDa luu ket qua CSV: {output_path.relative_to(Path(__file__).resolve().parent)}")
    print(f"Da tao bao cao HTML: {report_path.relative_to(Path(__file__).resolve().parent)}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
