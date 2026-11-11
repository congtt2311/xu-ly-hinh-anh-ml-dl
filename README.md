# GoPro Motion Deblurring

NAFNet is the primary model for restoring motion-blurred images. Wiener Filter
and DeblurGAN are comparison baselines. The project includes GoPro data
preparation, model training and evaluation, a Streamlit app, and HTML reports.

## 1. Dataset

Extract GoPro Large under `data/raw/GOPRO_Large/`. The project manifest contains
3,214 aligned blur/sharp pairs:

| Split | Pairs |
|---|---:|
| Train | 1,923 |
| Validation | 180 |
| Test | 1,111 |

Create or refresh the manifest with:

```powershell
python data\scripts\prepare_gopro.py
```

The script writes `data/processed/gopro_manifest.csv`. Keep the test split
separate from training.

## 2. Models and Evaluation

- **NAFNet:** primary deep-learning model, trained on GoPro.
- **Wiener Filter:** classical image-processing baseline.
- **DeblurGAN:** deep-learning baseline.

Models are compared on the same GoPro test images using PSNR, SSIM, and
inference time. NAFNet and DeblurGAN require their respective trained
checkpoints before they can be evaluated.

## 3. Project Structure

```text
.
├── app.py                         # Streamlit image-restoration app
├── data/
│   ├── raw/GOPRO_Large/           # Extracted dataset (not tracked by Git)
│   ├── processed/                 # Dataset manifests
│   └── scripts/                   # Dataset preparation and analysis
├── docs/
│   ├── description/               # Project and algorithm documentation
│   ├── images/                    # Evaluation charts
│   └── report/                    # Multi-page HTML report
├── models/algorithms/             # Wiener Filter, NAFNet, and DeblurGAN
├── outputs/                       # Existing evaluation CSV and charts
├── results/                       # Generated demo and evaluation outputs
├── scripts/                       # GoPro training and comparison evaluation
├── src/                           # Shared data, inference, and metric utilities
├── train.py                       # Supporting CNN training pipeline
├── evaluate.py                    # Supporting CNN evaluation pipeline
├── generate_report.py             # HTML report generator
└── xu-ly-hinh-anh-ml-dl/           # Additional ML/DL experiment code
```

## 4. Installation

From the repository root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

NAFNet uses BasicSR's NAFNet implementation. BasicSR is not currently listed
in `requirements.txt`; install it separately in a compatible environment before
training or evaluating NAFNet.

## 5. Train NAFNet

After preparing the GoPro manifest, train NAFNet with:

```powershell
python scripts\train_nafnet.py --epochs 100 --device auto
```

The best checkpoint is saved to `weights/NAFNet-GoPro-width32.pth`. Training
may take substantial time and memory; adjust `--batch-size` and `--crop-size`
to suit the available hardware.

## 6. Demo and Model Comparison

Launch the interactive app:

```powershell
python -m streamlit run app.py
```

Create a quick visual demo using three GoPro test images and the Wiener
baseline:

```powershell
python scripts\evaluate_deblurring.py --demo
```

The HTML gallery and comparison images are saved under `results/demo/`.

After training NAFNet and DeblurGAN, evaluate all three methods:

```powershell
python scripts\evaluate_deblurring.py --method both
```

Use `--method nafnet`, `--method deblurgan`, or `--method wiener` to run a
single method. Evaluation results are written under `results/`.

## 7. HTML Report

Generate the multi-page report in `docs/report/`:

```powershell
python generate_report.py
```

Open `docs/report/index.html` in a browser. The report distinguishes NAFNet
from the Wiener and DeblurGAN baselines; missing checkpoints or evaluation
results are reported as not evaluated. The legacy `outputs/evaluation_results.csv`
contains CNN + Wiener experiment data and should not be interpreted as NAFNet
results.
