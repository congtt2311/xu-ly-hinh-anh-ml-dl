# ĐẶC TẢ BÀI THỰC HÀNH: HỆ THỐNG KHÔI PHỤC VÀ LÀM NÉT ẢNH

## 1. Project Name
Hệ thống Khôi phục và Làm nét ảnh mờ Chuyển động (Motion Deblurring System)

## 2. Problem
Khôi phục chất lượng hình ảnh bị mờ do chuyển động (Motion Blur), chuyển đổi ảnh mờ thành ảnh rõ nét và đánh giá định lượng hiệu năng giữa thuật toán Học máy cổ điển và các mô hình Học sâu hiện đại.
- **Dữ liệu thực nghiệm:** Bộ dữ liệu chuẩn **GoPro Benchmark** gồm 3.214 cặp ảnh mờ / ảnh nét thực tế (`data/GoPro/` phân chia thành 2.103 cặp huấn luyện và 1.111 cặp kiểm thử).
- **Phạm vi bài lab:** Nghiên cứu lý thuyết, triển khai thuật toán lọc Wiener cổ điển, nạp và thực thi suy luận với 2 mô hình Deep Learning (DeblurGAN-v2, NAFNet), đo đạc chỉ số chất lượng/tốc độ và xây dựng giao diện người dùng trực quan.

## 3. Target User
- Giảng viên và sinh viên bộ môn Thị giác máy tính / Xử lý ảnh (đánh giá, so sánh hiệu năng giữa phương pháp xử lý tần số cổ điển và mô hình Deep Learning SOTA).
- Người dùng cuối hoặc nhà phát triển cần công cụ khôi phục ảnh nhòe/mờ chuyển động và nâng cao chất lượng hình ảnh theo thời gian thực.

## 4. Input / Output
- **Input:** Một ảnh đơn hoặc tập hợp ảnh đầu vào bị mờ do chuyển động (`Image_Blur`).
- **Output:** Ảnh đã được làm nét (`Image_Deblurred`) kèm bảng thống kê các chỉ số chất lượng định lượng (**PSNR**, **SSIM**) và thời gian xử lý (**Latency** tính bằng miligiây/giây).

## 5. Core Features
- **Khôi phục đa phương pháp:** Tích hợp 3 thuật toán từ cổ điển đến SOTA: Wiener Filter (Machine Learning / FFT), DeblurGAN-v2 (cGAN với FPN & PatchGAN), và NAFNet (Kiến trúc phi hàm kích hoạt với SimpleGate & SCA).
- **Đánh giá định lượng tự động:** Hệ thống tự động tính toán các chỉ số PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index) và Latency đối chiếu trực tiếp với ảnh gốc (Ground Truth).
- **Phân bổ mã nguồn & Tài liệu mô tả:**
  - `src/train_ml.py`: Chứa hàm `wiener_filter()` & `deblur_wiener_image()`.
  - `src/dl_models.py`: Chứa class `DeblurGANv2Predictor` & `NAFNetPredictor`.
  - `src/evaluate_metrics.py`: Chứa hàm tính `calculate_psnr()` & `calculate_ssim()`.
  - `docs/description/algorithms/`: Lưu trữ tài liệu giải thích chi tiết toán học cho từng thuật toán (`Wiener_Filter.MD`, `DeblurGAN.MD`, `NAFNet.MD`).
- **Demo giao diện tương tác (Gradio UI):** Ứng dụng web (`demo/app.py`) cho phép tải ảnh lên, chọn mô hình xử lý, và so sánh ảnh mờ / ảnh sau khôi phục Side-by-side trực quan.

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python 3.10+, PyTorch, OpenCV, NumPy, SciPy, Scikit-image, Gradio (xem chi tiết tại `requirements.txt`).
- **Cấu trúc lưu trữ:** Trọng số mô hình nạp sẵn lưu tại `weights/` (`.pth`).
- **Thực thi & Triển khai:** 
  - Khởi chạy ứng dụng Web Demo:
    ```bash
    python demo/app.py
    ```
  - Cập nhật mã nguồn và tài liệu qua Git:
    ```bash
    git add .
    git commit -m "Update project algorithms and documentation"
    git push origin main
    ```

## 7. Algorithm / Evaluation

### A. Thuật toán Lõi (Core Algorithms)
1. **Wiener Filter (Machine Learning Cổ điển):**
   - *Nguyên lý:* Xử lý trên miền tần số (FFT), ước lượng hàm truyền mờ $H(u,v)$ (PSF) và tối thiểu hóa bình phương sai số trung bình (MSE).
   - *Công thức bộ lọc:*
     $$\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + K} \right] G(u,v)$$
   - *Vai trò:* Làm Baseline mốc so sánh cổ điển với Deep Learning.

2. **DeblurGAN-v2 (Deep Learning - cGAN):**
   - *Kiến trúc:* Generator dạng **FPN (Feature Pyramid Network)** + Critic dạng **PatchGAN**.
   - *Cơ chế:* Học ánh xạ $I_B \rightarrow I_S$ với kết nối tàn dư toàn cục: $\hat{I}_S = I_B + I_R$
   - *Hàm tổn thất (Loss Function):*
     $$\mathcal{L}_{total} = \mathcal{L}_{GAN} + 100 \cdot \mathcal{L}_{X}$$
     *(Với $\mathcal{L}_{GAN}$ là WGAN-GP Loss và $\mathcal{L}_{X}$ là Perceptual Loss trích xuất từ lớp `conv3_3` của VGG-19).*

3. **NAFNet (Deep Learning - SOTA Model):**
   - *Đặc trưng cốt lõi:* Loại bỏ các hàm kích hoạt phi tuyến (ReLU/GELU), thay bằng **SimpleGate**:
     $$\text{SimpleGate}(X, Y) = X \odot Y$$
   - *Cải tiến:* Sử dụng **Simplified Channel Attention (SCA)** giúp tối ưu tốc độ và đạt chỉ số PSNR/SSIM vượt trội trên bộ GoPro.

### B. Đánh giá & Kiểm định (Evaluation)
- **Bộ dữ liệu kiểm thử:** Evaluated trên 1.111 cặp ảnh thuộc tập Test của bộ GoPro Benchmark.
- **Chỉ số đánh giá:** 
  - **PSNR (dB):** Đo lường tỷ lệ tín hiệu trên nhiễu đỉnh.
  - **SSIM:** Đo độ tương đồng cấu trúc thị giác giữa ảnh khôi phục và ảnh gốc.
  - **Latency (ms/frame):** Đo tốc độ suy luận thời gian thực trên GPU/CPU.

---

## Tài liệu Tham khảo

1. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018 (Phần lý thuyết Wiener Filter & Restoration in Frequency Domain).
2. S. Nah, T. H. Kim, K. M. Lee, *Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring*, CVPR 2017 (Bộ dữ liệu GoPro Benchmark).
3. O. Kupyn et al., *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks*, CVPR 2018. https://arxiv.org/abs/1711.07064
4. O. Kupyn et al., *DeblurGAN-v2: Deblurring (Resnet) Mobile Networks for Fast Image Deblurring*, ICCV 2019. https://arxiv.org/abs/1908.03828
5. L. Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022 (Mô hình NAFNet SOTA). https://arxiv.org/abs/2204.04676
6. I. Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017 (WGAN-GP Loss).
7. P. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017 (pix2pix, PatchGAN Architecture).