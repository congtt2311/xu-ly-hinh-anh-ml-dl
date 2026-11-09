
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "GOPRO_Large" / "GOPRO_Large"

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

CHECKPOINT_PATH = ROOT / "checkpoints"
OUTPUT_PATH = ROOT / "outputs"

IMAGE_SIZE = 128

BATCH_SIZE = 1

EPOCHS = 5

LEARNING_RATE = 0.0001

WIENER_SIZE = 5

NUM_WORKERS = 0

DEVICE = "cpu"
