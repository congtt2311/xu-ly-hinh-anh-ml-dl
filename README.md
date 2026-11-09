
# CNN + Wiener Filter for Image Deblurring

## 1. Introduction

This project develops a hybrid image deblurring system using a Wiener Filter and a Convolutional Neural Network (CNN).

The system aims to restore sharp images from blurred input images.

## 2. Dataset

Dataset: GoPro Large Dataset

- Training images: 2,103 pairs
- Testing images: 1,111 pairs
- Blur: Blurred input images
- Sharp: Ground-truth sharp images

Dataset structure:

data/GOPRO_Large/GOPRO_Large/
- train/
- test/

## 3. Proposed Method

The proposed system combines two methods:

### Wiener Filter
A traditional image processing technique used to reduce noise and improve image quality.

### CNN
A deep learning model that learns image restoration from paired blurred and sharp images.

The CNN receives:
- Original blurred image
- Wiener-filtered image

The CNN predicts a correction to restore the image.

## 4. Model Training

- Framework: PyTorch
- Model: CNN with Residual Blocks
- Loss Function: L1 Loss
- Optimizer: Adam
- Learning Rate: 0.0001
- Training device: CPU or CUDA when available

## 5. Evaluation Metrics

Two metrics are used:

- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index

## 6. Experimental Results

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Blur | 26.0938 | 0.7631 |
| Wiener | 25.1743 | 0.7264 |
| CNN + Wiener | 26.1177 | 0.7637 |

Evaluation uses 128x128 center crops from the test images.

## 7. Project Structure

- train.py: Train the CNN model
- evaluate.py: Evaluate model performance
- predict.py: Restore an input image
- visualize.py: Generate comparison charts
- app.py: Streamlit demonstration
- src/: Model, dataset, Wiener Filter and metrics
- checkpoints/: Saved model weights
- outputs/: Evaluation results and charts

## 8. Installation

Install dependencies:

```bash
py -3.11 -m pip install -r requirements.txt
```

## 9. Run the Project

Train:

```bash
py -3.11 train.py --epochs 3 --batch-size 1 --patch-size 128 --limit 200
```

Evaluate:

```bash
py -3.11 evaluate.py --limit 0 --patch-size 128
```

Generate charts:

```bash
py -3.11 visualize.py
```

Launch demo:

```bash
py -3.11 -m streamlit run app.py
```

## 10. Conclusion

The CNN + Wiener model achieved slightly higher PSNR and SSIM than the original blurred input in this experiment.

However, the improvement is small. Further training and model optimization are needed to improve deblurring performance.
