from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "news_classifier.joblib"

RANDOM_STATE = 42
VAL_SIZE = 0.2

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Science/Technology"}

