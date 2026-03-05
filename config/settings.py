"""Application settings loaded from environment and defaults."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, ASSETS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Model Configuration
WALL_DETECTION_MODEL = str(MODELS_DIR / "wall_detector_model.h5")
OBJECT_DETECTION_MODEL = str(MODELS_DIR / "object_detector_model.pb")
CONFIDENCE_THRESHOLD = 0.85
MIN_WALL_LENGTH = 10  # pixels
MIN_ROOM_AREA = 100  # pixels²

# Image Processing
TARGET_DPI = 150
TARGET_RESOLUTION = (2048, 2048)
PIXEL_TO_METER = 0.01  # 1 pixel = 0.01 meter

# 3D Model
DEFAULT_WALL_HEIGHT = 3.0  # meters
DEFAULT_WALL_THICKNESS = 0.2  # meters
DOOR_HEIGHT = 2.1  # meters
DOOR_WIDTH = 0.9  # meters
WINDOW_HEIGHT = 1.2  # meters
WINDOW_WIDTH = 1.5  # meters

# Rendering
ENABLE_VR = os.getenv("ENABLE_VR", "false").lower() == "true"
RENDER_ENGINE = os.getenv("RENDER_ENGINE", "unity")  # "unity", "unreal", "web"
TEXTURE_QUALITY = "high"  # "low", "medium", "high"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "reconstruct.log"
