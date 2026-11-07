import cv2
import numpy as np
import time

def wiener_filter(img, psf, K=0.01):
    """
    Áp dụng thuật toán Wiener Filter trên miền tần số.
    :param img: Ảnh đầu vào (Grayscale hoặc kênh đơn)
    :param psf: Point Spread Function (Hàm truyền mờ)
    :param K: Tỷ lệ hằng số Noise-to-Signal Power Ratio
    :return: Ảnh đã làm nét
    """
    # 1. Chuyển ảnh và PSF sang miền tần số bằng FFT
    img_fft = np.fft.fft2(img)
    psf_fft = np.fft.fft2(psf, s=img.shape)
    
    # 2. Tính toán Wiener Filter Transfer Function: H* / (|H|^2 + K)
    psf_fft_conj = np.conj(psf_fft)
    kernel = psf_fft_conj / (np.abs(psf_fft)**2 + K)
    
    # 3. Phép nhân trên miền tần số và khôi phục về miền không gian bằng IFFT
    result_fft = img_fft * kernel
    result = np.abs(np.fft.ifft2(result_fft))
    
    # Chuẩn hóa giá trị pixel về [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def deblur_wiener_image(image_path, kernel_size=15, K=0.01):
    """
    Hàm đọc ảnh, xử lý Wiener Filter và đo thời gian thực thi.
    """
    start_time = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
        
    # Tạo Motion Blur Kernel giả định (PSF)
    psf = np.zeros((kernel_size, kernel_size))
    psf[int((kernel_size - 1) / 2), :] = 1.0 / kernel_size

    # Nếu là ảnh màu, xử lý độc lập từng kênh R, G, B
    if len(img.shape) == 3:
        channels = cv2.split(img)
        restored_channels = []
        for c in channels:
            restored_c = wiener_filter(c, psf, K)
            restored_channels.append(restored_c)
        restored_img = cv2.merge(restored_channels)
    else:
        restored_img = wiener_filter(img, psf, K)

    process_time = time.time() - start_time
    return restored_img, process_time

if __name__ == "__main__":
    # Chạy thử nghiệm file
    # image_result, elapsed_time = deblur_wiener_image("data/GoPro/test/input/0.png")
    # print(f"Thời gian xử lý: {elapsed_time:.4f} giây")
    pass
