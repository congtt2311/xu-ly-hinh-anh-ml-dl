
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "raw" / "GOPRO_Large"

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
MANIFEST_PATH = ROOT / "data" / "processed" / "gopro_manifest.csv"

CHECKPOINT_PATH = ROOT / "checkpoints"
OUTPUT_PATH = ROOT / "outputs"
IMAGE_OUTPUT_PATH = ROOT / "docs" / "images"
REPORT_PATH = ROOT / "docs" / "report"

IMAGE_SIZE = 128

BATCH_SIZE = 1

EPOCHS = 5

LEARNING_RATE = 0.0001

WIENER_SIZE = 5

NUM_WORKERS = 0

DEVICE = "cpu"
