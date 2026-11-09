
import argparse
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split, Subset

from config import (
    TRAIN_PATH,
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WIENER_SIZE,
    NUM_WORKERS
)

from src.dataset import GoProDataset
from src.model import CNNWienerDeblur
from src.engine import train_one_epoch, validate


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_dataset = GoProDataset(
        TRAIN_PATH,
        patch_size=args.patch_size,
        training=True
    )

    val_dataset = GoProDataset(
        TRAIN_PATH,
        patch_size=args.patch_size,
        training=False
    )

    if args.limit > 0:
        limit = min(args.limit, len(train_dataset))

        train_dataset.pairs = train_dataset.pairs[:limit]
        val_dataset.pairs = val_dataset.pairs[:limit]

    if len(train_dataset) < 2:
        raise RuntimeError(
            "Can it nhat 2 cap anh de huan luyen."
        )

    val_count = max(
        1,
        int(len(train_dataset) * 0.1)
    )

    train_count = len(train_dataset) - val_count

    train_split, val_split = random_split(
        range(len(train_dataset)),
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42)
    )

    train_data = Subset(
        train_dataset,
        train_split.indices
    )

    val_data = Subset(
        val_dataset,
        val_split.indices
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_data,
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
            "num_blocks": 8
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
