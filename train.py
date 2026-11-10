
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset

from config import (
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WIENER_SIZE,
    NUM_WORKERS,
    MANIFEST_PATH,
)

from scripts.gopro_dataset import GoProDeblurDataset
from src.model import CNNWienerDeblur
from src.engine import train_one_epoch, validate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.patch_size < 1:
        parser.error("--patch-size must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_dataset = GoProDeblurDataset(
        "train",
        manifest_path=MANIFEST_PATH,
        crop_size=args.patch_size,
    )

    val_dataset = GoProDeblurDataset(
        "valid",
        manifest_path=MANIFEST_PATH,
    )

    if args.limit > 0:
        train_dataset = Subset(
            train_dataset,
            range(min(args.limit, len(train_dataset))),
        )
        val_dataset = Subset(
            val_dataset,
            range(min(args.limit, len(val_dataset))),
        )

    if not len(train_dataset) or not len(val_dataset):
        raise RuntimeError(
            "Training and validation splits must both contain image pairs."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = CNNWienerDeblur().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    CHECKPOINT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            WIENER_SIZE
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            WIENER_SIZE
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Validation Loss: {val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "wiener_size": WIENER_SIZE,
            "features": 64,
            "num_blocks": 8,
            "seed": args.seed,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(val_dataset),
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH / "last_model.pth"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                checkpoint,
                CHECKPOINT_PATH / "best_model.pth"
            )

            print("Saved best_model.pth")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
