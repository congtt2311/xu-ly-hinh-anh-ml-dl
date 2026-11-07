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

## Exploratory data analysis

After the processed splits are available, run:

```powershell
python data\scripts\eda.py
```

The script validates the CSV-to-manifest mappings and LR/HR files, then writes
summary CSVs, sample image statistics, and plots to `data/processed/eda`.
Generated analysis artifacts are excluded from Git with the processed data.
