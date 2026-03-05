"""Pytest fixtures and shared test data."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_binary_blueprint():
    """Minimal binary image: white walls on black (simple L-shape)."""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:55, 50:150] = 255   # horizontal wall
    img[50:150, 145:150] = 255  # vertical wall
    return img


@pytest.fixture
def sample_grayscale():
    """Grayscale 100x100 image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def temp_blueprint_path(tmp_path, sample_binary_blueprint):
    """Save sample blueprint to temp file and return path."""
    import cv2
    p = tmp_path / "blueprint.png"
    cv2.imwrite(str(p), sample_binary_blueprint)
    return str(p)
