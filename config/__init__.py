"""Configuration package for ReconStruct."""

from config.paths import (
    ASSETS_3D,
    ASSETS_MATERIALS,
    ASSETS_TEXTURES,
    DATA_DIR,
    DATA_INPUT,
    DATA_INTERMEDIATE,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ensure_dirs,
)
from config.settings import (
    BASE_DIR,
    CONFIDENCE_THRESHOLD,
    DEFAULT_WALL_HEIGHT,
    DEFAULT_WALL_THICKNESS,
    DOOR_HEIGHT,
    DOOR_WIDTH,
    ENABLE_VR,
    LOG_LEVEL,
    LOG_FILE,
    MIN_ROOM_AREA,
    MIN_WALL_LENGTH,
    OBJECT_DETECTION_MODEL,
    PIXEL_TO_METER,
    RENDER_ENGINE,
    TARGET_DPI,
    TARGET_RESOLUTION,
    TEXTURE_QUALITY,
    WALL_DETECTION_MODEL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from config.constants import (
    DOOR_LABEL,
    EXPORT_GLB,
    EXPORT_GLTF,
    EXPORT_OBJ,
    WALL_LABEL,
    WINDOW_LABEL,
    SUPPORTED_IMAGE_EXTENSIONS,
)

__all__ = [
    "ensure_dirs",
    "paths",
    "settings",
    "constants",
]
