from pathlib import Path

ROOT = Path(__file__).parent

SAMPLES_DIR = ROOT / "samples"
RAW_DATA = SAMPLES_DIR / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
BEST_MODELS_DIR = ROOT / "data" / "best_models"


def processed_path(model_name: str) -> Path:
    return PROCESSED_DIR / model_name
