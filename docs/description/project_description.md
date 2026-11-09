# TÓM TẮT DỰ ÁN: HỆ THỐNG KHÔI PHỤC VÀ LÀM NÉT ẢNH

## 1. Mục tiêu Dự án
- Khôi phục ảnh mờ chuyển động (Motion Deblurring) bằng **Machine Learning** (Wiener Filter) và **Deep Learning** (DeblurGAN-v2, NAFNet).
- Đánh giá định lượng trên bộ dữ liệu **GoPro Benchmark** qua các chỉ số **PSNR, SSIM, Latency**.
- Cung cấp giao diện trực quan **Gradio UI** cho phép so sánh ảnh và các chỉ số trực tiếp.

---

## 2. Cấu trúc Thư mục Dự án (`Deblurring_Project/`)

```text
Deblurring_Project/
├── data/GoPro/               # Tập train (2103 cặp) & test (1111 cặp)
├── src/                      # Mã nguồn xử lý chính (.py)
│   ├── train_ml.py           # Thuật toán Wiener Filter
│   ├── dl_models.py          # Class nạp mô hình DeblurGAN-v2 & NAFNet
│   └── evaluate_metrics.py   # Hàm tính chỉ số PSNR, SSIM, Latency
├── docs/description/         # Tài liệu mô tả dự án (.md)
│   ├── algorithms/           # File mô tả chi tiết các thuật toán
│   │   ├── Wiener_Filter.MD
│   │   ├── DeblurGAN.MD
│   │   └── NAFNet.MD
│   └── project_description.md# Tổng quan dự án
├── demo/app.py               # Giao diện Web demo bằng Gradio
├── weights/                  # File trọng số pretrained (.pth)
└── requirements.txt          # Khai báo các thư viện phụ thuộc
---
```
## 3. Tóm tắt các Thuật toán Lõi

### A. Wiener Filter (Machine Learning Cổ điển)
- **Nguyên lý:** Xử lý trên miền tần số (FFT), ước lượng hàm truyền mờ $H(u,v)$ (PSF) và tối thiểu hóa bình phương sai số trung bình (MSE).
- **Công thức bộ lọc:**
$$\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + K} \right] G(u,v)$$
- **Vai trò:** Làm Baseline mốc so sánh cổ điển với Deep Learning.

### B. DeblurGAN-v2 (Deep Learning - cGAN)
- **Kiến trúc:** Generator dạng **FPN (Feature Pyramid Network)** + Critic dạng **PatchGAN**.
- **Cơ chế:** Học ánh xạ $I_B \rightarrow I_S$ với kết nối tàn dư toàn cục:
$$\hat{I}_S = I_B + I_R$$
- **Hàm tổn thất (Loss Function):**
$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + 100 \cdot \mathcal{L}_{X}$$
*(Với $\mathcal{L}_{GAN}$ là WGAN-GP Loss và $\mathcal{L}_{X}$ là Perceptual Loss trích xuất từ lớp `conv3_3` của VGG-19).*

### C. NAFNet (Deep Learning - SOTA Model)
- **Đặc trưng cốt lõi:** Loại bỏ các hàm kích hoạt phi tuyến (ReLU/GELU), thay bằng **SimpleGate**:
$$\text{SimpleGate}(X, Y) = X \odot Y$$
- **Cải tiến:** Sử dụng **Simplified Channel Attention (SCA)** giúp tối ưu tốc độ và đạt chỉ số PSNR/SSIM vượt trội trên bộ GoPro.

---
```
## 4. Phân bổ File Mã nguồn & Tài liệu
```
| Tên File | Thư mục | Nội dung chính |
|---|---|---|
| `train_ml.py` | `src/` | Chứa hàm `wiener_filter()` & `deblur_wiener_image()` |
| `dl_models.py` | `src/` | Chứa class `DeblurGANv2Predictor` & `NAFNetPredictor` |
| `evaluate_metrics.py` | `src/` | Chứa hàm tính `calculate_psnr()` & `calculate_ssim()` |
| `app.py` | `demo/` | Giao diện Gradio tương tác so sánh ảnh Side-by-side |
| `DeblurGAN.MD` | `docs/description/algorithms/` | Tài liệu toán học chi tiết cho mô hình DeblurGAN |

---

## 5. Các lệnh Git cơ bản

```bash
git add .
git commit -m "Update project algorithms and documentation"
git push origin main

---

```
## Tham khảo

1. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018 (Phần lý thuyết Wiener Filter & Restoration in Frequency Domain).
2. S. Nah, T. H. Kim, K. M. Lee, *Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring*, CVPR 2017 (Bộ dữ liệu GoPro Benchmark).
3. O. Kupyn et al., *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks*, CVPR 2018. https://arxiv.org/abs/1711.07064
4. O. Kupyn et al., *DeblurGAN-v2: Deblurring (Resnet) Mobile Networks for Fast Image Deblurring*, ICCV 2019. https://arxiv.org/abs/1908.03828
5. L. Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022 (Mô hình NAFNet SOTA). https://arxiv.org/abs/2204.04676
6. I. Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017 (WGAN-GP Loss).
7. P. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017 (pix2pix, PatchGAN Architecture).
