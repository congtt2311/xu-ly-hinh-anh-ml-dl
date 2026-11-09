"""Train a NAFNet deblurrer on the GoPro dataset."""

from __future__ import annotations

import argparse
import importlib
import math
import random
from pathlib import Path

import numpy as np
try:
    torch = importlib.import_module("torch")
except ImportError as exc:
    raise ImportError(
        "NAFNet training requires PyTorch. Install it in the active Python environment."
    ) from exc
functional = torch.nn.functional

if __package__:
    from .gopro_dataset import build_gopro_dataloader
else:
    from gopro_dataset import build_gopro_dataloader


ROOT = Path(__file__).resolve().parents[1]


def make_model(device: torch.device) -> torch.nn.Module:
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError as exc:
        raise ImportError(
            "NAFNet training requires BasicSR. Install the project requirements "
            "before starting training."
        ) from exc

    return NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    ).to(device)


def predict(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    height, width = inputs.shape[-2:]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        inputs = functional.pad(inputs, (0, pad_width, 0, pad_height), mode="reflect")
    return model(inputs)[..., :height, :width]


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = predict(model, inputs).clamp(0, 1)
            total_loss += functional.l1_loss(outputs, targets).item()
            mse = functional.mse_loss(outputs, targets).item()
            total_psnr += math.inf if mse == 0 else 10 * math.log10(1 / mse)
    count = len(loader)
    return total_loss / count, total_psnr / count


def train(args: argparse.Namespace) -> None:
    if args.crop_size % 8:
        raise ValueError("--crop-size must be divisible by 8 for NAFNet")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader = build_gopro_dataloader(
        "train",
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        manifest_path=args.manifest,
    )
    valid_loader = build_gopro_dataloader(
        "valid",
        batch_size=1,
        crop_size=None,
        num_workers=args.num_workers,
        manifest_path=args.manifest,
    )
    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf

    print(f"Device: {device}")
    print(f"Training pairs: {len(train_loader.dataset)}")
    print(f"Validation pairs: {len(valid_loader.dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.l1_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        validation_loss, validation_psnr = evaluate(model, valid_loader, device)
        training_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_l1={training_loss:.6f} "
            f"valid_l1={validation_loss:.6f} "
            f"valid_psnr={validation_psnr:.2f} dB"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "params": model.state_dict(),
                    "epoch": epoch,
                    "valid_l1": validation_loss,
                    "valid_psnr": validation_psnr,
                },
                output_path,
            )
            print(f"Saved best checkpoint: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "gopro_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "weights" / "NAFNet-GoPro-width32.pth",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    try:
        train(args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
