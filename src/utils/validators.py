"""Input validation for blueprint paths and images."""

from pathlib import Path
from typing import Tuple

import numpy as np

from config.constants import SUPPORTED_IMAGE_EXTENSIONS


def validate_blueprint_path(path: str) -> Tuple[bool, str]:
    """
    Validate that path exists and has a supported image extension.
    Returns (success, error_message).
    """
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    if not p.is_file():
        return False, f"Not a file: {path}"
    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"Unsupported extension: {p.suffix}. Use one of {SUPPORTED_IMAGE_EXTENSIONS}"
    return True, ""


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validate numpy image array (non-empty, 2D or 3D)."""
    if image is None:
        return False, "Image is None"
    if not isinstance(image, np.ndarray):
        return False, "Image must be numpy array"
    if image.size == 0:
        return False, "Image is empty"
    if image.ndim not in (2, 3):
        return False, "Image must be 2D (grayscale) or 3D (BGR/RGB)"
    return True, ""
