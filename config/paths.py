"""Path configuration for ReconStruct pipeline."""

from pathlib import Path

# Base paths
_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE_DIR / "data"
MODELS_DIR = _BASE_DIR / "models"
ASSETS_DIR = _BASE_DIR / "assets"
OUTPUT_DIR = _BASE_DIR / "output"
LOGS_DIR = _BASE_DIR / "logs"

# Asset subdirectories
ASSETS_3D = ASSETS_DIR / "3d_objects"
ASSETS_TEXTURES = ASSETS_DIR / "textures"
ASSETS_MATERIALS = ASSETS_DIR / "materials"

# Data subdirectories
DATA_INPUT = DATA_DIR / "input"
DATA_INTERMEDIATE = DATA_DIR / "intermediate"


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for directory in [
        DATA_DIR,
        DATA_INPUT,
        DATA_INTERMEDIATE,
        MODELS_DIR,
        ASSETS_DIR,
        ASSETS_3D,
        ASSETS_TEXTURES,
        ASSETS_MATERIALS,
        OUTPUT_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
