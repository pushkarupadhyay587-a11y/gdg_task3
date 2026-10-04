from pathlib import Path

from src.config import LABELS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectoriser.pkl"
LEGACY_MODEL_PATH = ARTIFACTS_DIR / "models" / "LinearSVC().pkl"

API_TITLE = "News Classifier API"
API_VERSION = "1.0.0"
