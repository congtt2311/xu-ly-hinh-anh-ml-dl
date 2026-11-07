import torch
import cv2
import numpy as np
import time
from torchvision import transforms

class DeblurGANv2Predictor:
    def __init__(self, weight_path="weights/deblurgan_fpn.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_path = weight_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Khởi tạo mô hình và tải trọng số Pretrained.
        (Lưu ý: Thay thế Generator bằng class DeblurGANv2 tương ứng với repo bạn sử dụng)
        """
        try:
            # Ví dụ tải model từ file weight PyTorch
            model = torch.load(self.weight_path, map_location=self.device)
            if isinstance(model, torch.nn.Module):
                model.eval()
                return model.to(self.device)
            else:
                # Nếu weight chỉ chứa state_dict
                # model = FPNInceptionv2()
                # model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
                pass
        except Exception as e:
            print(f"[Warning] Chưa tìm thấy weight tại {self.weight_path} hoặc sai cấu trúc: {e}")
            return None

    def predict(self, image_input):
        """
        Nhận vào ảnh (đường dẫn str hoặc numpy array BGR) và trả về ảnh đã làm nét + thời gian chạy.
        """
        start_time = time.time()
        
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        # Tiền xử lý: Normalize về [-1, 1] hoặc [0, 1] tùy mô hình
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor_img = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model is not None:
                output_tensor = self.model(tensor_img)
            else:
                # Mockup output nếu chưa nạp file weight .pth thật
                output_tensor = tensor_img

        # Hậu xử lý: Chuyển Tensor về lại ảnh Numpy uint8
        output_img = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_img = ((output_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        process_time = time.time() - start_time
        return output_img, process_time

if __name__ == "__main__":
    predictor = DeblurGANv2Predictor()
    print("Khởi tạo DeblurGAN-v2 predictor thành công!")
