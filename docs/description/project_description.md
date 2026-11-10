# ĐẶC TẢ BÀI THỰC HÀNH: HỆ THỐNG KHÔI PHỤC VÀ LÀM NÉT ẢNH

## 1. Project Name
Hệ thống Khôi phục và Làm nét ảnh mờ Chuyển động (Motion Deblurring System)

## 2. Problem
Khôi phục chất lượng hình ảnh bị mờ do chuyển động (Motion Blur), chuyển đổi ảnh mờ thành ảnh rõ nét và đánh giá định lượng hiệu năng giữa thuật toán Học máy cổ điển và các mô hình Học sâu hiện đại.
- **Dữ liệu thực nghiệm:** GoPro Large gồm 3.214 cặp ảnh trong manifest: 1.923 train, 180 validation và 1.111 test. Ảnh nằm trong `data/raw/GOPRO_Large/`; manifest ở `data/processed/gopro_manifest.csv`.
- **Phạm vi triển khai chính:** Wiener Filter kết hợp CNN residual, huấn luyện bằng PyTorch, đánh giá PSNR/SSIM và giao diện Streamlit. DeblurGAN và NAFNet có mã nguồn riêng, không phải mô hình mặc định trong luồng train/evaluate này.

## 3. Target User
- Giảng viên và sinh viên bộ môn Thị giác máy tính / Xử lý ảnh (đánh giá, so sánh hiệu năng giữa phương pháp xử lý tần số cổ điển và mô hình Deep Learning SOTA).
- Người dùng cuối hoặc nhà phát triển cần công cụ khôi phục ảnh nhòe/mờ chuyển động và nâng cao chất lượng hình ảnh theo thời gian thực.

## 4. Input / Output
- **Input:** Một ảnh đơn hoặc tập hợp ảnh đầu vào bị mờ do chuyển động (`Image_Blur`).
- **Output:** Ảnh đã được làm nét (`Image_Deblurred`) kèm bảng thống kê các chỉ số chất lượng định lượng (**PSNR**, **SSIM**) và thời gian xử lý (**Latency** tính bằng miligiây/giây).

## 5. Core Features
- **Khôi phục ảnh:** CNN residual nhận ảnh blur cùng ảnh Wiener để dự đoán phần hiệu chỉnh; Wiener Filter cũng được đánh giá như baseline.
- **Đánh giá định lượng tự động:** Hệ thống tự động tính toán các chỉ số PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index) và Latency đối chiếu trực tiếp với ảnh gốc (Ground Truth).
- **Phân bổ mã nguồn & Tài liệu mô tả:**
  - `src/wiener.py`, `src/model.py`, `src/engine.py`: lọc Wiener, CNN residual và vòng lặp train/validation.
  - `src/metrics.py`: các chỉ số PSNR và SSIM.
  - `evaluate.py`, `predict.py`, `app.py`: đánh giá, phục hồi ảnh đơn và giao diện Streamlit.
  - `docs/description/algorithms/`: tài liệu về Wiener Filter, DeblurGAN và NAFNet.
- **Demo giao diện tương tác (Streamlit):** `app.py` cho phép tải ảnh mờ, xem kết quả phục hồi, tải ảnh đầu ra và đánh giá nếu có ảnh sharp tham chiếu.

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python 3.10+, PyTorch, OpenCV, NumPy, SciPy, Scikit-image và Streamlit (xem `requirements.txt`).
- **Cấu trúc lưu trữ:** Checkpoint CNN lưu tại `checkpoints/`; trọng số cho các mô hình bổ sung lưu theo hướng dẫn tương ứng.
- **Thực thi & Triển khai:**
  - Khởi chạy ứng dụng Web Demo:
    ```bash
    streamlit run app.py
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
- **Bộ dữ liệu kiểm thử:** Manifest hiện có 1.111 cặp ảnh thuộc tập test GoPro. Có thể dùng `python evaluate.py --limit 0` để đánh giá toàn bộ khi đã huấn luyện CNN.
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