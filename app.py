
import os
import io
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import CHECKPOINT_PATH
from src.inference import load_model, restore_image
from src.metrics import evaluate_metrics


st.set_page_config(
    page_title="CNN + Wiener Filter",
    page_icon="🖼️",
    layout="wide"
)


@st.cache_resource
def get_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = CHECKPOINT_PATH / "best_model.pth"

    model, wiener_size = load_model(
        model_path,
        device
    )

    return model, wiener_size, device


def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def image_to_bytes(image):
    image = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(image).save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def create_comparison_image(images, labels):

    tile_width = 600
    tile_height = 400
    title_height = 50
    margin = 20

    canvas_width = tile_width * 2 + margin * 3

    canvas_height = (
        (tile_height + title_height) * 2
        + margin * 3
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24
        )
    except OSError:
        font = ImageFont.load_default()

    positions = [
        (margin, margin),
        (tile_width + margin * 2, margin),
        (margin, tile_height + title_height + margin * 2),
        (
            tile_width + margin * 2,
            tile_height + title_height + margin * 2
        )
    ]

    for image_array, label, position in zip(
        images,
        labels,
        positions
    ):

        image_array = np.clip(
            image_array * 255.0,
            0,
            255
        ).astype(np.uint8)

        image = Image.fromarray(
            image_array
        ).convert("RGB")

        image = ImageOps.contain(
            image,
            (tile_width, tile_height)
        )

        x, y = position

        draw.text(
            (x, y),
            label,
            fill="black",
            font=font
        )

        image_x = (
            x + (tile_width - image.width) // 2
        )

        image_y = y + title_height

        canvas.paste(
            image,
            (image_x, image_y)
        )

    return canvas


def calculate_image_metrics(
    original,
    wiener_image,
    restored,
    sharp
):

    blur_metrics = evaluate_metrics(
        original,
        sharp
    )

    wiener_metrics = evaluate_metrics(
        wiener_image,
        sharp
    )

    restored_metrics = evaluate_metrics(
        restored,
        sharp
    )

    return (
        blur_metrics,
        wiener_metrics,
        restored_metrics
    )


st.title("CNN + Wiener Filter")

st.subheader(
    "GoPro Large Image Deblurring System"
)

st.write(
    "Ứng dụng khôi phục ảnh bị nhòe bằng cách kết hợp "
    "Wiener Filter và Convolutional Neural Network."
)

st.divider()


checkpoint_path = CHECKPOINT_PATH / "best_model.pth"

if not checkpoint_path.exists():

    st.error(
        "Không tìm thấy best_model.pth. "
        "Hãy huấn luyện mô hình trước."
    )

    st.stop()


uploaded_blur = st.file_uploader(
    "Chọn ảnh bị nhòe (Blur Image)",
    type=["png", "jpg", "jpeg"]
)

uploaded_sharp = st.file_uploader(
    "Chọn ảnh Sharp chuẩn để đánh giá (không bắt buộc)",
    type=["png", "jpg", "jpeg"],
    key="sharp"
)


if st.button(
    "Khôi phục ảnh",
    type="primary",
    disabled=uploaded_blur is None
):

    temp_path = None

    try:

        blur_array = read_uploaded_image(
            uploaded_blur
        )

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

        Image.fromarray(
            blur_array
        ).save(temp_path)

        model, wiener_size, device = get_model()

        with st.spinner("Đang khôi phục ảnh..."):

            start_time = time.time()

            original, wiener_image, restored = restore_image(
                temp_path,
                model,
                device,
                wiener_size
            )

            processing_time = time.time() - start_time

        st.success(
            "Khôi phục ảnh thành công!"
        )

        st.write(
            f"**Thiết bị:** {device} | "
            f"**Thời gian xử lý:** {processing_time:.3f} giây"
        )

        st.subheader("Kết quả khôi phục ảnh")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                original,
                caption="Blur - Ảnh đầu vào",
                use_container_width=True
            )

        with col2:

            st.image(
                wiener_image,
                caption="Wiener Filter",
                use_container_width=True
            )

        with col3:

            st.image(
                restored,
                caption="CNN + Wiener",
                use_container_width=True
            )

        st.download_button(
            label="Tải ảnh đã khôi phục",
            data=image_to_bytes(restored),
            file_name="cnn_wiener_restored.png",
            mime="image/png"
        )

        if uploaded_sharp is not None:

            sharp_array = read_uploaded_image(
                uploaded_sharp
            )

            if sharp_array.shape[:2] != restored.shape[:2]:

                sharp_array = cv2.resize(
                    sharp_array,
                    (
                        restored.shape[1],
                        restored.shape[0]
                    ),
                    interpolation=cv2.INTER_AREA
                )

            sharp_array = (
                sharp_array.astype(np.float32) / 255.0
            )

            (
                blur_metrics,
                wiener_metrics,
                restored_metrics
            ) = calculate_image_metrics(
                original,
                wiener_image,
                restored,
                sharp_array
            )

            st.divider()

            st.subheader(
                "Đánh giá chất lượng ảnh"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:

                st.metric(
                    "Blur PSNR",
                    f"{blur_metrics['PSNR']:.4f} dB"
                )

                st.metric(
                    "Blur SSIM",
                    f"{blur_metrics['SSIM']:.4f}"
                )

            with metric_col2:

                st.metric(
                    "Wiener PSNR",
                    f"{wiener_metrics['PSNR']:.4f} dB"
                )

                st.metric(
                    "Wiener SSIM",
                    f"{wiener_metrics['SSIM']:.4f}"
                )

            with metric_col3:

                st.metric(
                    "CNN + Wiener PSNR",
                    f"{restored_metrics['PSNR']:.4f} dB"
                )

                st.metric(
                    "CNN + Wiener SSIM",
                    f"{restored_metrics['SSIM']:.4f}"
                )

            st.divider()

            st.subheader(
                "Ảnh Sharp chuẩn"
            )

            st.image(
                sharp_array,
                caption="Sharp - Ground Truth",
                use_container_width=True
            )

            comparison_image = create_comparison_image(
                [
                    original,
                    wiener_image,
                    restored,
                    sharp_array
                ],
                [
                    "1. Blur",
                    "2. Wiener Filter",
                    "3. CNN + Wiener",
                    "4. Sharp Ground Truth"
                ]
            )

            st.divider()

            st.subheader(
                "So sánh kết quả 4 phương pháp"
            )

            st.image(
                comparison_image,
                use_container_width=True
            )

            comparison_buffer = io.BytesIO()

            comparison_image.save(
                comparison_buffer,
                format="PNG"
            )

            st.download_button(
                label="Tải ảnh so sánh 4 phương pháp",
                data=comparison_buffer.getvalue(),
                file_name="comparison_cnn_wiener.png",
                mime="image/png"
            )

            st.divider()

            st.subheader(
                "Biểu đồ so sánh chất lượng"
            )

            chart_data = pd.DataFrame(
                {
                    "PSNR (dB)": [
                        float(blur_metrics["PSNR"]),
                        float(wiener_metrics["PSNR"]),
                        float(restored_metrics["PSNR"])
                    ],
                    "SSIM": [
                        float(blur_metrics["SSIM"]),
                        float(wiener_metrics["SSIM"]),
                        float(restored_metrics["SSIM"])
                    ]
                },
                index=[
                    "Blur",
                    "Wiener",
                    "CNN + Wiener"
                ]
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.write("PSNR (dB)")

                st.bar_chart(
                    chart_data["PSNR (dB)"]
                )

            with chart_col2:

                st.write("SSIM")

                st.bar_chart(
                    chart_data["SSIM"]
                )

    except Exception as error:

        st.error(
            f"Có lỗi xảy ra: {error}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


st.divider()

st.caption(
    "CNN + Wiener Filter | GoPro Large Dataset | "
    "PyTorch | OpenCV | Streamlit"
)
