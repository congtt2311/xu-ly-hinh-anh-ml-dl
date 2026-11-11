# ĐẶC TẢ BÀI THỰC HÀNH: HỆ THỐNG KHÔI PHỤC VÀ LÀM NÉT ẢNH

## 1. Project Name
Hệ thống Khôi phục và Làm nét ảnh mờ Chuyển động (Motion Deblurring System)

## 2. Problem
Khôi phục chất lượng hình ảnh bị mờ do chuyển động (Motion Blur), chuyển đổi ảnh mờ thành ảnh rõ nét và đánh giá định lượng hiệu năng giữa thuật toán Học máy cổ điển và các mô hình Học sâu hiện đại.
- **Dữ liệu thực nghiệm:** GoPro Large gồm 3.214 cặp ảnh trong manifest: 1.923 train, 180 validation và 1.111 test. Ảnh nằm trong `data/raw/GOPRO_Large/`; manifest ở `data/processed/gopro_manifest.csv`.
- **Mô hình chính:** NAFNet huấn luyện trên GoPro để khử mờ chuyển động. Wiener Filter (cổ điển) và DeblurGAN (học sâu) là hai baseline dùng để so sánh.

## 3. Target User
- Giảng viên và sinh viên bộ môn Thị giác máy tính / Xử lý ảnh cần trình bày pipeline và so sánh phương pháp khử mờ.
- Người dùng thử nghiệm muốn khôi phục ảnh đơn bằng giao diện hiện có hoặc chạy script đánh giá trên GoPro.

## 4. Input / Output
- **Input:** Một ảnh đơn hoặc tập hợp ảnh đầu vào bị mờ do chuyển động (`Image_Blur`).
- **Output:** Ảnh đã được làm nét (`Image_Deblurred`) kèm bảng thống kê các chỉ số chất lượng định lượng (**PSNR**, **SSIM**) và thời gian xử lý (**Latency** tính bằng miligiây/giây).

## 5. Core Features
- **Khôi phục đa phương pháp:** NAFNet là mô hình trọng tâm; Wiener Filter và DeblurGAN là các mốc so sánh độc lập.
- **Đánh giá định lượng tự động:** Hệ thống tự động tính toán các chỉ số PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index) và Latency đối chiếu trực tiếp với ảnh gốc (Ground Truth).
- **Phân bổ mã nguồn & Tài liệu mô tả:**
  - `scripts/train_nafnet.py`: huấn luyện NAFNet trên GoPro.
  - `scripts/evaluate_deblurring.py`: so sánh NAFNet, Wiener và DeblurGAN trên cùng tập test.
  - `src/wiener.py`, `src/model.py`, `src/engine.py`: Wiener Filter và pipeline CNN residual cũ, độc lập với NAFNet.
  - `src/metrics.py`: các chỉ số PSNR và SSIM.
  - `evaluate.py`, `predict.py`, `app.py`: đánh giá, phục hồi ảnh đơn và giao diện Streamlit.
  - `docs/description/algorithms/`: tài liệu về Wiener Filter, DeblurGAN và NAFNet.
- **Demo:** `app.py` là giao diện Streamlit của pipeline CNN + Wiener cũ. Demo khử mờ NAFNet và các baseline được chạy bằng `scripts/evaluate_deblurring.py`; hai luồng chưa được tích hợp chung vào Streamlit.

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python, PyTorch, BasicSR, OpenCV, NumPy, SciPy, Scikit-image và Streamlit (xem `requirements.txt`; cần kiểm tra tương thích phiên bản trong môi trường cài đặt).
- **Cấu trúc lưu trữ:** Checkpoint NAFNet lưu tại `weights/NAFNet-GoPro-width32.pth`; checkpoint các baseline lưu theo cấu hình tương ứng.
- **Thực thi & Triển khai:**
  - Khởi chạy giao diện Streamlit của pipeline CNN + Wiener cũ:
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

2. **DeblurGAN (baseline học sâu):**
   - *Kiến trúc trong mã nguồn:* Generator ResNet 9 residual blocks và critic PatchGAN; đây không phải kiến trúc FPN của DeblurGAN-v2.
   - *Cơ chế:* Học ánh xạ $I_B \rightarrow I_S$ với kết nối tàn dư toàn cục: $\hat{I}_S = I_B + I_R$
   - *Hàm tổn thất (Loss Function):*
     $$\mathcal{L}_{total} = \mathcal{L}_{GAN} + 100 \cdot \mathcal{L}_{X}$$
     *(Với $\mathcal{L}_{GAN}$ là WGAN-GP Loss và $\mathcal{L}_{X}$ là Perceptual Loss trích xuất từ lớp `conv3_3` của VGG-19).*

3. **NAFNet (mô hình học sâu chính):**
   - *Đặc trưng cốt lõi:* Loại bỏ các hàm kích hoạt phi tuyến (ReLU/GELU), thay bằng **SimpleGate**:
     $$\text{SimpleGate}(X, Y) = X \odot Y$$
   - *Đặc trưng:* Sử dụng **Simplified Channel Attention (SCA)**. Chất lượng của cấu hình trong đồ án cần được xác nhận bằng checkpoint và đánh giá thực nghiệm; không suy ra thành tích từ kết quả đã công bố cho cấu hình khác.

### B. Đánh giá & Kiểm định (Evaluation)
- **Bộ dữ liệu kiểm thử:** Manifest hiện có 1.111 cặp ảnh thuộc tập test GoPro. Dùng `python scripts/evaluate_deblurring.py --method both` để chạy Wiener, NAFNet và DeblurGAN khi đã có checkpoint tương ứng cho hai mô hình học sâu.
- **Chỉ số đánh giá:**
  - **PSNR (dB):** Đo lường tỷ lệ tín hiệu trên nhiễu đỉnh.
  - **SSIM:** Đo độ tương đồng cấu trúc thị giác giữa ảnh khôi phục và ảnh gốc.
  - **Latency (ms/frame):** Đo thời gian suy luận trên thiết bị được báo cáo; không mặc định xem kết quả là thời gian thực.

---

## Tài liệu Tham khảo

1. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018 (Phần lý thuyết Wiener Filter & Restoration in Frequency Domain).
2. S. Nah, T. H. Kim, K. M. Lee, *Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring*, CVPR 2017 (Bộ dữ liệu GoPro Benchmark).
3. O. Kupyn et al., *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks*, CVPR 2018. https://arxiv.org/abs/1711.07064
4. O. Kupyn et al., *DeblurGAN-v2: Deblurring (Resnet) Mobile Networks for Fast Image Deblurring*, ICCV 2019. https://arxiv.org/abs/1908.03828
5. L. Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022 (Mô hình NAFNet SOTA). https://arxiv.org/abs/2204.04676
6. I. Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017 (WGAN-GP Loss).
7. P. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017 (pix2pix, PatchGAN Architecture).