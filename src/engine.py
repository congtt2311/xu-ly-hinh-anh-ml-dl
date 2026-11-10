
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .wiener import batch_wiener


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["input"], batch["target"]
    return batch


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    wiener_size=5
):

    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training"
    )

    for batch in progress:
        blur, sharp = _unpack_batch(batch)

        blur = blur.to(device)
        sharp = sharp.to(device)

        wiener = batch_wiener(
            blur,
            size=wiener_size
        )

        optimizer.zero_grad(set_to_none=True)

        output = model(
            blur,
            wiener
        )

        loss = F.l1_loss(
            output,
            sharp
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    wiener_size=5
):

    model.eval()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Validation"
    ):
        blur, sharp = _unpack_batch(batch)

        blur = blur.to(device)
        sharp = sharp.to(device)

        wiener = batch_wiener(
            blur,
            size=wiener_size
        )

        output = model(
            blur,
            wiener
        )

        loss = F.l1_loss(
            output,
            sharp
        )

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
