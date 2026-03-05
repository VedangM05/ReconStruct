"""Tests for wall and object detection."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processing.wall_detector import WallDetector
from src.image_processing.object_detector import ObjectDetector


def test_wall_detector_hough(sample_binary_blueprint):
    det = WallDetector(min_wall_length=10)
    segments = det.detect_lines_hough(sample_binary_blueprint)
    assert isinstance(segments, np.ndarray)
    assert segments.ndim == 2 and segments.shape[1] == 4


def test_wall_detector_merge():
    det = WallDetector(min_wall_length=5)
    segments = np.array([[0, 0, 50, 0], [52, 0, 100, 0]], dtype=float)
    merged = det.merge_collinear(segments, angle_tol=0.1, dist_tol=5.0)
    assert len(merged) >= 1
    for m in merged:
        assert len(m) == 4


def test_object_detector_contours(sample_binary_blueprint):
    det = ObjectDetector()
    # Add a small rectangle (door-like)
    img = sample_binary_blueprint.copy()
    img[80:120, 80:100] = 255
    objects = det.detect_contours_rectangles(img, min_area=50)
    assert isinstance(objects, list)


def test_object_detector_detect(sample_binary_blueprint):
    det = ObjectDetector()
    objects = det.detect(sample_binary_blueprint, binary=sample_binary_blueprint, use_ml=False)
    assert isinstance(objects, list)
