"""Tests for image preprocessing."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processing.preprocessor import BlueprintPreprocessor


def test_preprocessor_resize():
    pre = BlueprintPreprocessor(target_resolution=(512, 512))
    img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    out = pre.resize(img)
    assert out.shape == (512, 512, 3)


def test_to_grayscale():
    pre = BlueprintPreprocessor()
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    gray = pre.to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)


def test_binarize():
    pre = BlueprintPreprocessor()
    gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    binary = pre.binarize(gray)
    assert binary.ndim == 2
    assert set(np.unique(binary)).issubset({0, 255})


def test_run_missing_file():
    pre = BlueprintPreprocessor()
    with pytest.raises(FileNotFoundError):
        pre.run("/nonexistent/image.png")


def test_run_with_real_file(temp_blueprint_path):
    pre = BlueprintPreprocessor(target_resolution=(256, 256))
    out = pre.run(temp_blueprint_path, denoise=False, binarize_output=True)
    assert "image" in out
    assert "gray" in out
    assert "binary" in out
    assert out["image"].shape[:2] == (256, 256)
    assert out["binary"].ndim == 2
