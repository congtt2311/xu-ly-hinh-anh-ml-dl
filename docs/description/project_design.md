# THIẾT KẾ DỰ ÁN — HỆ THỐNG KHỬ MỜ ẢNH CHUYỂN ĐỘNG

## BÀI THỰC HÀNH XỬ LÝ ẢNH VÀ HỌC SÂU

---

### 1. WHAT — DỰ ÁN LÀM GÌ?

- **Bài toán:** Khôi phục ảnh rõ từ ảnh bị mờ do chuyển động (motion deblurring)
  bằng mô hình học sâu và phương pháp xử lý ảnh cổ điển.
- **Đầu vào / đầu ra:**
  - *Đầu vào:* Ảnh RGB bị mờ (`blur`) và, khi đánh giá, ảnh rõ tham chiếu
    (`sharp`) tương ứng.
  - *Đầu ra:* Ảnh đã khôi phục cùng các chỉ số PSNR, SSIM và thời gian suy luận.
- **Dữ liệu thực nghiệm:** GoPro Large, manifest hiện có 3.214 cặp ảnh:
  1.923 cặp train, 180 cặp validation và 1.111 cặp test. Ảnh được tổ chức theo
  chuỗi cảnh; manifest tại `data/processed/gopro_manifest.csv` trỏ đến ảnh trong
  `data/raw/GOPRO_Large/`.
- **Mô hình và vai trò:**
  - **NAFNet:** mô hình chính của đề tài, huấn luyện trên GoPro.
  - **Wiener Filter:** baseline xử lý ảnh cổ điển.
  - **DeblurGAN:** baseline học sâu để so sánh.
- **Phạm vi:**
  - *Có làm:* Chuẩn bị manifest dữ liệu, huấn luyện NAFNet, đánh giá các phương
    pháp trên cùng tập test, tạo demo ảnh và báo cáo HTML.
  - *Không làm:* Triển khai dịch vụ production, xử lý video thời gian thực,
    huấn luyện từ dữ liệu ngoài phạm vi GoPro hoặc cam kết chất lượng chưa được
    đo bằng thực nghiệm.
- **Kết quả mong đợi:** Pipeline có thể tái chạy, kết quả so sánh được lưu rõ
  nguồn gốc và báo cáo phân biệt chính xác số liệu của từng phương pháp. Không
  xem số liệu CNN + Wiener trong thí nghiệm cũ là kết quả NAFNet.

---

### 2. WHY — TẠI SAO CHỌN BÀI TOÁN VÀ CÁCH TIẾP CẬN NÀY?

- **Ý nghĩa bài toán:** Nhòe chuyển động làm mất chi tiết và giảm khả năng quan
  sát của ảnh. Khôi phục ảnh là bài toán tiêu biểu trong xử lý ảnh và thị giác
  máy tính.
- **Vì sao dùng dữ liệu GoPro:** Bộ dữ liệu cung cấp cặp ảnh mờ/rõ có căn chỉnh,
  phù hợp cho huấn luyện có giám sát và đánh giá trực tiếp trên ảnh tham chiếu.
  Việc tách train, validation và test giúp dành riêng test cho đánh giá cuối.
- **Vì sao chọn NAFNet làm mô hình chính:** NAFNet là kiến trúc phục hồi ảnh
  được thiết kế để cân bằng chất lượng và hiệu quả tính toán. Thiết lập trong
  dự án dùng NAFNet width 32, 12 middle blocks, encoder `[2, 2, 4, 8]` và
  decoder `[2, 2, 2, 2]`.
- **Vì sao có hai baseline:**
  - Wiener Filter cung cấp mốc cổ điển đơn giản, dựa trên giả định kernel mờ
    Gaussian cố định; giả định này không mô tả đầy đủ chuyển động mờ biến thiên
    trong GoPro.
  - DeblurGAN cung cấp mốc học sâu khác với NAFNet; generator ResNet có kết nối
    residual toàn cục, critic PatchGAN và loss WGAN-GP kết hợp perceptual loss.
- **Vì sao dùng PSNR, SSIM và latency:** PSNR đo sai số tái tạo theo cường độ
  pixel; SSIM phản ánh độ tương đồng cấu trúc; latency cho biết thời gian xử lý.
  Cần xem đồng thời các chỉ số thay vì kết luận chỉ từ một thước đo.
- **Mục tiêu học tập:** Thực hành đầy đủ quy trình xử lý dữ liệu ảnh, huấn
  luyện, đánh giá định lượng, trực quan hóa và trình bày kết quả có thể kiểm
  chứng.

---

### 3. WHICH / WHEN / WHERE — DÙNG GÌ, THEO TRÌNH TỰ NÀO, Ở ĐÂU?

#### WHICH — Công cụ, thuật toán và tiêu chí đánh giá

| Thành phần | Lựa chọn trong dự án | Vai trò |
|---|---|---|
| Ngôn ngữ và tính toán | Python, NumPy, PyTorch | Xử lý dữ liệu và chạy mô hình |
| Đọc và ghi ảnh | Pillow, OpenCV | Nạp cặp ảnh, chuyển đổi màu và xuất kết quả |
| Mô hình chính | NAFNet qua BasicSR | Huấn luyện và khử mờ ảnh GoPro |
| Baseline cổ điển | Wiener Filter | So sánh phương pháp miền tần số |
| Baseline học sâu | DeblurGAN | So sánh với một mô hình GAN phục hồi ảnh |
| Chỉ số | PSNR, SSIM, inference time | Đo chất lượng và tốc độ xử lý |
| Giao diện và báo cáo | Streamlit, HTML | Demo ứng dụng và trình bày kết quả |

- **Tiêu chí so sánh:** Chạy các phương pháp trên cùng các ảnh test và cặp ảnh
  tham chiếu; báo cáo số ảnh, PSNR, SSIM và latency theo từng phương pháp.
- **Tiêu chí chọn kết luận:** Phân tích chất lượng ảnh, các chỉ số định lượng
  và thời gian suy luận; không khẳng định mô hình tốt hơn nếu chưa có kết quả
  đánh giá tương ứng.

#### WHEN — Các giai đoạn thực hiện

1. **Chuẩn bị dữ liệu:** Kiểm tra cấu trúc GoPro, tạo manifest và xác nhận các
   cặp blur/sharp theo split.
2. **Huấn luyện mô hình chính:** Nạp split train và validation, tạo crop huấn
   luyện 256 × 256, tối ưu NAFNet bằng AdamW với L1 loss và lưu checkpoint có
   validation loss tốt nhất.
3. **Chuẩn bị mốc so sánh:** Nạp Wiener Filter và checkpoint DeblurGAN nếu có.
   Phương pháp không có checkpoint không được ghi nhận như đã đánh giá.
4. **Đánh giá:** Chạy trên split test, tính PSNR/SSIM và thời gian suy luận,
   lưu CSV kết quả theo ảnh và bảng tổng hợp.
5. **Trình bày:** Tạo gallery HTML cho demo ảnh, sinh báo cáo nhiều trang và
   xem lại nguồn gốc số liệu trước khi trình bày.

#### WHERE — Cấu trúc mã nguồn và dữ liệu

```text
.
├── app.py                              # Giao diện Streamlit hiện có
├── config.py                           # Đường dẫn và cấu hình chung
├── data/
│   ├── raw/GOPRO_Large/                # Ảnh GoPro gốc, không đưa lên Git
│   ├── processed/gopro_manifest.csv    # Danh sách cặp ảnh và split
│   └── scripts/
│       ├── prepare_gopro.py            # Tạo manifest GoPro
│       └── eda.py                      # Khảo sát dữ liệu
├── docs/
│   ├── description/                    # Mô tả, thiết kế và thuật toán
│   ├── images/                         # Biểu đồ đánh giá
│   └── report/                         # Báo cáo HTML nhiều trang
├── models/
│   ├── algorithms/                     # NAFNet, DeblurGAN, Wiener Filter
│   └── saved_models/deblurgan/         # Checkpoint DeblurGAN
├── scripts/
│   ├── gopro_dataset.py                # Dataset và dataloader GoPro
│   ├── train_nafnet.py                 # Huấn luyện mô hình chính
│   └── evaluate_deblurring.py          # So sánh và tạo demo HTML
├── src/                                # Pipeline CNN cũ, inference và metrics
├── weights/                            # Checkpoint NAFNet
├── outputs/evaluation_results.csv      # Kết quả thí nghiệm CNN + Wiener cũ
└── results/                            # CSV đánh giá và gallery demo sinh ra
```

---

### 4. HOW — LUỒNG XỬ LÝ TỪ ĐẦU VÀO ĐẾN ĐẦU RA

#### Bước 1: Chuẩn bị dữ liệu

- **Đầu vào:** GoPro Large đã giải nén dưới `data/raw/GOPRO_Large/`.
- **Xử lý:** `data/scripts/prepare_gopro.py` lập manifest cặp ảnh blur/sharp,
  giữ các split train, validation và test riêng biệt.
- **Đầu ra:** `data/processed/gopro_manifest.csv`, không sao chép lại ảnh gốc.

#### Bước 2: Huấn luyện NAFNet

- **Đầu vào:** Các cặp ảnh thuộc split train và validation trong manifest.
- **Xử lý:** Dataset nạp ảnh RGB về `[0, 1]`; crop ngẫu nhiên đồng bộ cho train.
  NAFNet tối ưu L1 loss bằng AdamW và đánh giá validation sau mỗi epoch.
- **Đầu ra:** Checkpoint tốt nhất tại
  `weights/NAFNet-GoPro-width32.pth`.

#### Bước 3: Khôi phục và so sánh mô hình

- **Đầu vào:** Ảnh blur thuộc split test; checkpoint NAFNet/DeblurGAN nếu chạy
  các mô hình cần trọng số.
- **Xử lý:** Chạy Wiener Filter, NAFNet và/hoặc DeblurGAN theo lựa chọn của
  `scripts/evaluate_deblurring.py`; đối chiếu kết quả với ảnh sharp cùng tên.
- **Đầu ra:** `results/gopro_test_metrics.csv` chứa kết quả từng ảnh và
  `results/gopro_test_metrics_summary.csv` chứa số liệu trung bình.

#### Bước 4: Tạo demo trực quan

- **Đầu vào:** Một số ảnh test blur/sharp và phương pháp được chọn.
- **Xử lý:** Chạy `scripts/evaluate_deblurring.py --demo`; chế độ demo mặc định
  dùng Wiener Filter và ba ảnh để có thể chạy trước khi huấn luyện mô hình sâu.
- **Đầu ra:** Ảnh so sánh trong `results/demo/images/` và trang
  `results/demo/index.html`.

#### Bước 5: Sinh báo cáo và trình bày

- **Đầu vào:** CSV đánh giá, manifest và trạng thái checkpoint.
- **Xử lý:** `generate_report.py` tạo các trang HTML về dữ liệu, pipeline,
  huấn luyện và đánh giá.
- **Đầu ra:** `docs/report/index.html` cùng các trang con trong
  `docs/report/`.

> **Lưu ý về phạm vi ứng dụng:** `app.py` hiện là giao diện Streamlit của pipeline
> CNN + Wiener cũ. Quy trình NAFNet và so sánh ba phương pháp hiện được chạy qua
> các script trong `scripts/`; không nên mô tả ứng dụng Streamlit hiện tại như
> giao diện NAFNet nếu chưa nối trực tiếp hai luồng này.

---

### 5. CÁCH CHẠY CÁC THÀNH PHẦN CHÍNH

Từ thư mục gốc của repository:

```powershell
# Tạo manifest GoPro
python data\scripts\prepare_gopro.py

# Huấn luyện NAFNet
python scripts\train_nafnet.py --epochs 100 --device auto

# Chạy demo nhanh bằng Wiener Filter
python scripts\evaluate_deblurring.py --demo

# So sánh các mô hình (cần checkpoint tương ứng cho NAFNet/DeblurGAN)
python scripts\evaluate_deblurring.py --method both

# Sinh báo cáo HTML
python generate_report.py
```

BasicSR cần có trong môi trường để huấn luyện hoặc đánh giá NAFNet. DeblurGAN
cũng cần checkpoint và các dependency huấn luyện tương ứng. Các checkpoint và
ảnh dữ liệu lớn không được lưu trong Git.

---

### Tài liệu tham khảo

1. S. Nah, T. H. Kim, K. M. Lee, “Deep Multi-scale Convolutional Neural Network
   for Dynamic Scene Deblurring,” CVPR 2017.
2. L. Chen et al., “Simple Baselines for Image Restoration,” ECCV 2022.
   https://arxiv.org/abs/2204.04676
3. O. Kupyn et al., “DeblurGAN: Blind Motion Deblurring Using Conditional
   Adversarial Networks,” CVPR 2018. https://arxiv.org/abs/1711.07064
4. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson,
   2018.