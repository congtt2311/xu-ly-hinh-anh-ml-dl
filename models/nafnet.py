import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class NAFNetDeblurrer:
    def __init__(self, weights_path: str, use_gpu: bool = True):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weights_path}")
        try:
            from basicsr.models.archs.NAFNet_arch import NAFNet
            from basicsr.utils import img2tensor, tensor2img
        except ImportError as exc:
            raise ImportError(
                "NAFNet cần BasicSR. Cài dependency này trước khi khởi tạo mô hình."
            ) from exc

        self.img2tensor = img2tensor
        self.tensor2img = tensor2img
        self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        print(f"[INFO] Khởi tạo NAFNet trên thiết bị: {self.device.type.upper()}")

        self.model = NAFNet(
            img_channel=3, width=32, middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]
        ).to(self.device)

        print(f"[INFO] Đang tải trọng số từ: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)

        if 'params' in checkpoint:
            self.model.load_state_dict(checkpoint['params'], strict=True)
        else:
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.eval()
        print("[INFO] Tải mô hình NAFNet thành công!")

    def deblur_image(self, image_path: str, output_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh tại: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).to(self.device)
        height, width = img_tensor.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        if pad_height or pad_width:
            img_tensor = torch.nn.functional.pad(
                img_tensor, (0, pad_width, 0, pad_height), mode="reflect"
            )

        with torch.no_grad():
            output_tensor = self.model(img_tensor)[..., :height, :width]

        output_img = self.tensor2img(output_tensor, rgb2bgr=True, min_max=(0, 1))
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        if not cv2.imwrite(output_path, output_img):
            raise OSError(f"Không thể lưu ảnh khử mờ tại: {output_path}")

        print(f"[INFO] Đã lưu ảnh khử mờ tại: {output_path}")
        return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

# =====================================================================
# SỬA LỖI 3: Cập nhật lại đường dẫn trong hàm Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAFNet on one blurry image.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "NAFNet-GoPro-width32.pth",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "nafnet_output" / "restored.png",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image not found: {args.input}")
    deblurrer = NAFNetDeblurrer(str(args.weights), use_gpu=not args.cpu)
    deblurrer.deblur_image(str(args.input), str(args.output))
