# Super-resolution dataset

The prepared Unsplash image pairs are stored in `data/processed`:

```text
data/processed/
├── train/
│   ├── LR/
│   └── HR/
├── valid/
│   ├── LR/
│   └── HR/
├── test/
│   ├── LR/
│   └── HR/
└── manifest.csv
```

Each LR image is paired with an HR copy bearing the same filename. `manifest.csv`
maps the processed paths to the original filenames. Splits are assigned by
source HR image, so its LR variants stay in the same split.

The current preparation has 3,762 pairs: 3,009 train, 375 validation, and 378
test. No source pairs were missing. `data/raw/image_data.csv` contains the
pairing metadata. The large source-image archive and extracted image folders
are not kept in `data/raw`; restore the `high res` and `low res` folders under
`data/raw/dataset/Image Super Resolution - Unsplash` before recreating the
processed images:

```powershell
python data\scripts\prepare_super_resolution.py
```

The script uses a fixed random seed (`42`) and leaves `data/raw` unchanged.
Processed image files are excluded from Git because of their size.

## GoPro Large deblurring dataset

The GoPro Large archive can be extracted to `data/raw/GOPRO_Large` without
replacing the Unsplash super-resolution data above. Its source layout is:

```text
data/raw/GOPRO_Large/
├── train/<sequence>/
│   ├── blur/
│   ├── blur_gamma/
│   └── sharp/
└── test/<sequence>/
    ├── blur/
    ├── blur_gamma/
    └── sharp/
```

Use the images under `blur` as deblurring inputs and the matching filenames
under `sharp` as targets. `blur_gamma` contains alternate blurry inputs. The
archive contains 2,103 training and 1,111 test frames per image category. Keep
the test sequences out of training and validation; if a validation split is
needed, make it from training sequences.

The raw dataset is excluded from Git by `.gitignore` because it requires about
9.5 GB when extracted. The ZIP archive is not copied into the repository.

Create a small manifest without copying the images:

```powershell
python data\scripts\prepare_gopro.py
```

The script holds out two complete training sequences for validation by default
(seed `42`) and keeps all official test sequences separate. It writes
`data/processed/gopro_manifest.csv`; the Unsplash `manifest.csv` and image
folders remain independent.

Load GoPro batches from the repository root with:

```python
from scripts.gopro_dataset import build_gopro_dataloader

train_loader = build_gopro_dataloader("train", batch_size=1, crop_size=256)
batch = next(iter(train_loader))
blurred, sharp = batch["input"], batch["target"]
```

Validation and test loaders use full-resolution images (set `crop_size=None`
for train to do the same). The images are returned as RGB float tensors in
`[0, 1]`; only training crops are randomized.

Install the project's dependencies, then train NAFNet from scratch:

```powershell
python -m pip install -r xu-ly-hinh-anh-ml-dl\requirements.txt
python scripts\train_nafnet.py --epochs 100 --device auto
```

The best checkpoint is saved to `weights/NAFNet-GoPro-width32.pth`. Checkpoint
and result files are excluded from Git. Training can take substantial time and
requires enough memory; use `--batch-size`, `--crop-size`, and `--num-workers`
to fit the available hardware.

Wiener Filter is the classical baseline, NAFNet is the primary model, and
DeblurGAN is an additional deep-learning comparison. Train the additional model
with its GoPro/WGAN-GP training command:

```powershell
python models\algorithms\DeblurGAN.py train --epochs 300
```

Its generator checkpoint is saved to
`models/saved_models/deblurgan/G_latest.pth`. VGG19 pretrained weights are
downloaded by torchvision for the perceptual loss on the first training run.

After training both deep models, compare all three on the same official GoPro
test images:

```powershell
python scripts\evaluate_deblurring.py
```

The evaluation reports per-image and mean PSNR, SSIM, and inference time to
`results/gopro_test_metrics.csv` and
`results/gopro_test_metrics_summary.csv`. To test only the Wiener baseline
before training, run
`python scripts\evaluate_deblurring.py --method wiener --limit 1`.
Wiener uses a fixed Gaussian blur-kernel assumption, which cannot fully model
GoPro's spatially varying motion blur.

## Exploratory data analysis

After the processed splits are available, run:

```powershell
python data\scripts\eda.py
```

The script validates the CSV-to-manifest mappings and LR/HR files, then writes
summary CSVs, sample image statistics, and plots to `data/processed/eda`.
Generated analysis artifacts are excluded from Git with the processed data.
