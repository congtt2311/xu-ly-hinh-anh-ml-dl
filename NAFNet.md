# NAFNet (Nonlinear Activation Free Network)

## 1. Mục tiêu
*   **Mục tiêu chung:** Xây dựng một kiến trúc mạng nơ-ron có độ phức tạp thấp nhưng đạt hiệu suất cao (State-of-the-Art - SOTA) trong bài toán phục hồi hình ảnh (Image Restoration), bao gồm khử mờ (Deblurring) và khử nhiễu (Denoising).
*   **Mục tiêu trong đồ án:** Đóng vai trò là mô hình Deep Learning cốt lõi để giải quyết bài toán "Khôi phục và làm nét ảnh mờ". NAFNet được chọn nhờ khả năng khôi phục chi tiết xuất sắc, tốc độ suy luận vượt trội so với các mô hình Transformer (như Restormer) và có sẵn trọng số huấn luyện (pretrained weights) trên chuẩn GoPro.

## 2. Đầu vào / Đầu ra
*   **Đầu vào (Input):** Ảnh bị mờ (Motion Blur hoặc Defocus Blur) dưới dạng tensor `(1, 3, H, W)`. Kích thước ảnh không thay đổi.
*   **Đầu ra (Output):** Ảnh đã được khử mờ, sắc nét dưới dạng tensor `(1, 3, H, W)`. Kích thước ảnh giữ nguyên so với đầu vào.

## 3. Evaluation Metric / Criteria
*   **PSNR (Peak Signal-to-Noise Ratio):** Đo sai số tuyệt đối giữa ảnh khôi phục và ảnh gốc (Ground Truth). Trên GoPro, NAFNet đạt xấp xỉ **33.69 dB** (bản width=64).
*   **SSIM (Structural Similarity):** Đo độ tương đồng về cấu trúc. Giá trị càng gần 1 càng tốt.
*   **Inference Time (Thời gian suy luận):** Đo thời gian xử lý trên 1 bức ảnh 1080p (tính bằng ms/giây).

## 4. How it work (Cách hoạt động)
NAFNet kế thừa kiến trúc U-Net (Encoder-Decoder) với các kết nối tắt (Skip Connections), nhưng có 2 điểm đột phá cốt lõi trong **NAFBlock** (Khối mạng chính):
1.  **Loại bỏ hàm kích hoạt (Nonlinear Activation Free):** Các nghiên cứu chỉ ra rằng các hàm kích hoạt như ReLU, GELU, Sigmoid là không cần thiết cho hiệu suất SOTA mà lại tốn tài nguyên. NAFNet thay thế bằng phép nhân phần tử đơn giản (**Simple Gate** - chia tensor làm 2 và nhân chúng với nhau).
2.  **Simplified Channel Attention (SCA):** Sử dụng một biến thể đơn giản của cơ chế Attention để giảm độ phức tạp tính toán.
*   **Quy trình:** Ảnh mờ đi qua Encoder (trích xuất đặc trưng) -> Đi qua các NAFBlock ở lớp giữa -> Đi qua Decoder (tái tạo ảnh) -> Kết hợp với ảnh gốc qua Skip Connection để cho ra ảnh sắc nét.

## 5. Data Used (Dữ liệu sử dụng)
*   **Dataset:** GoPro (Bộ dữ liệu chuẩn cho bài toán khử mờ chuyển động).
*   **Cấu trúc:** 2,103 cặp ảnh dùng để Train và 1,111 cặp ảnh dùng để Test. Mỗi cặp bao gồm 1 ảnh Input (bị mờ) và 1 ảnh Target (sắc nét).

## 6. Tài liệu tham khảo
*   Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). *Simple Baselines for Image Restoration*. ECCV 2022.
*   Repository chính thức: `https://github.com/megvii-research/NAFNet`