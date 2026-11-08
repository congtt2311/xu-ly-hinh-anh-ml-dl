import torch
import cv2
import numpy as np
import os
import sys

# =====================================================================
# SỬA LỖI 1: Tự động xác định thư mục gốc của dự án (Project Root)
# Giúp code chạy đúng dù bạn đứng ở thư mục nào trong Terminal
# =====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Nếu file nafnet.py nằm trong thư mục con (vd: src/), cần lùi lại 1 cấp
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
sys.path.append(PROJECT_ROOT)

# Import thư viện NAFNet (Yêu cầu đã cài đặt BasicSR)
try:
    from basicsr.models.archs.NAFNet_arch import NAFNet
    from basicsr.utils import img2tensor, tensor2img
except ImportError:
    print("[ERROR] Chưa cài đặt thư viện. Hãy chạy: pip install basicsr")

class NAFNetDeblurrer:
    def __init__(self, weights_path: str, use_gpu: bool = True):
        self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        print(f"[INFO] Khởi tạo NAFNet trên thiết bị: {self.device.type.upper()}")
        
        self.model = NAFNet(
            img_channel=3, width=32, middle_blk_num=12, 
            enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]
        ).to(self.device)
        
        # SỬA LỖI 2: Kiểm tra đường dẫn tuyệt đối (Absolute Path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weights_path}")
            
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
        img_tensor = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
            
        output_img = tensor2img(output_tensor, rgb2bgr=True, min_max=(0, 1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)
        
        print(f"[INFO] Đã lưu ảnh khử mờ tại: {output_path}")
        return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

# =====================================================================
# SỬA LỖI 3: Cập nhật lại đường dẫn trong hàm Main
# =====================================================================
if __name__ == "__main__":
    # Sử dụng đường dẫn tuyệt đối từ gốc dự án để tránh lỗi "File not found"
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "NAFNet-GoPro-width32.pth")
    INPUT_IMG = os.path.join(PROJECT_ROOT, "data", "GoPro", "test", "input", "blur_1.png")
    OUTPUT_IMG = os.path.join(PROJECT_ROOT, "results", "nafnet_output", "sharpen_1.png")
    
    # Khởi tạo và chạy
    deblurrer = NAFNetDeblurrer(weights_path=WEIGHTS_PATH, use_gpu=True)
    
    if os.path.exists(INPUT_IMG):
        deblurrer.deblur_image(INPUT_IMG, OUTPUT_IMG)
    else:
        print(f"[WARNING] Không tìm thấy ảnh test tại {INPUT_IMG}.")
