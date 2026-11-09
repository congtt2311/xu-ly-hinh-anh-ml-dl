
import argparse
from pathlib import Path

import torch

from config import CHECKPOINT_PATH, OUTPUT_PATH
from src.inference import load_model, restore_image, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = CHECKPOINT_PATH / "best_model.pth"

    model, wiener_size = load_model(checkpoint_path, device)

    blur, wiener, restored = restore_image(
        args.input,
        model,
        device,
        wiener_size
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(blur, output_dir / "blur.png")
    save_image(wiener, output_dir / "wiener.png")
    save_image(restored, output_dir / "restored.png")

    print("Da phuc hoi anh thanh cong!")
    print(f"Anh Blur: {output_dir / 'blur.png'}")
    print(f"Anh Wiener: {output_dir / 'wiener.png'}")
    print(f"Anh CNN: {output_dir / 'restored.png'}")


if __name__ == "__main__":
    main()
