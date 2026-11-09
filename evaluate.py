
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TEST_PATH, CHECKPOINT_PATH, WIENER_SIZE, NUM_WORKERS
from src.dataset import GoProDataset
from src.model import CNNWienerDeblur
from src.wiener import batch_wiener
from src.metrics import evaluate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=128)
    args = parser.parse_args()

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

    dataset = GoProDataset(
        TEST_PATH,
        patch_size=args.patch_size,
        training=False
    )

    if args.limit > 0:
        dataset.pairs = dataset.pairs[:args.limit]

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
        for blur, sharp in tqdm(loader, desc="Evaluating"):
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

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "evaluation_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Method", "PSNR", "SSIM"]
        )
        writer.writeheader()
        writer.writerows(final_results)

    print(f"\nDa luu ket qua CSV: {output_path}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
