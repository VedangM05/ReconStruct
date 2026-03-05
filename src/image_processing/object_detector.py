"""Detect doors, windows, and other objects in blueprint images."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from config import settings
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from config.constants import DOOR_LABEL, WINDOW_LABEL, FIXTURE_LABEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedObject:
    """Single detected object (door, window, etc.)."""
    label: str
    bbox: tuple  # (x_min, y_min, x_max, y_max) pixels
    confidence: float
    center: tuple = None  # (cx, cy)

    def __post_init__(self):
        if self.center is None and self.bbox:
            x1, y1, x2, y2 = self.bbox
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


class ObjectDetector:
    """Detect doors, windows, fixtures in blueprint images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = None,
    ):
        self.model_path = model_path or settings.OBJECT_DETECTION_MODEL
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self._model = None
        self.classes = ["door", "window"]

    def load_model(self):
        """Load pre-trained object detection model (TensorFlow SavedModel or placeholder)."""
        if self._model is not None:
            return
        path = Path(self.model_path)
        try:
            import tensorflow as tf
            self._model = tf.saved_model.load(str(path) if path.is_dir() else str(path.parent))
            logger.info("Loaded TensorFlow model from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load TensorFlow model: %s", e)
            logger.info("Using placeholder detection model")
            self._model = None

    def _load_model(self):
        """Lazy load TF SavedModel or frozen graph if available."""
        self.load_model()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        preprocessed = cv2.resize(image, (640, 640))
        preprocessed = preprocessed.astype(np.float32) / 255.0
        return np.expand_dims(preprocessed, axis=0)

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect doors and windows in image.
        Returns list of {"class": "door"|"window", "bbox": (x1,y1,x2,y2), "confidence": float}.
        """
        self.load_model()
        if self._model is None:
            return self._detect_objects_heuristic(image)
        try:
            preprocessed = self.preprocess_image(image)
            predictions = self._model(preprocessed)
            detections = self._parse_predictions(predictions, image.shape)
            logger.info("Detected %d objects", len(detections))
            return detections
        except Exception as e:
            logger.error("Detection failed: %s", e)
            return self._detect_objects_heuristic(image)

    def _parse_predictions(self, predictions, image_shape) -> List[Dict[str, Any]]:
        """Parse model output to structured detections (placeholder)."""
        return []

    def _detect_objects_heuristic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback heuristic detection using edge/contour analysis.
        Identifies rectangular regions that could be doors or windows.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.0 and 100 < area < 100000:
                obj_class = "door" if area > 2000 else "window"
                confidence = min(0.7, area / 10000)
                detections.append({
                    "class": obj_class,
                    "bbox": (x, y, x + w, y + h),
                    "confidence": confidence,
                })
        logger.info("Heuristic detection found %d objects", len(detections))
        return detections

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        confidence_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Filter detections by confidence threshold."""
        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        filtered = [d for d in detections if d["confidence"] >= thresh]
        logger.info(
            "Filtered to %d detections (threshold: %s)", len(filtered), thresh
        )
        return filtered

    def visualize_detections(
        self, image: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Visualize detected objects on image."""
        viz = image.copy()
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            obj_class = detection["class"]
            confidence = detection["confidence"]
            color = (255, 0, 0) if obj_class == "door" else (0, 255, 0)
            cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{obj_class}: {confidence:.2f}"
            cv2.putText(
                viz, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
        return viz

    def detect_contours_rectangles(
        self,
        binary: np.ndarray,
        min_area: int = 50,
        max_area_ratio: float = 0.2,
    ) -> List[DetectedObject]:
        """
        Heuristic: find rectangular contours as door/window candidates.
        Label by aspect ratio and size (door ~ tall, window ~ wide).
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h, w = binary.shape[:2]
        total_area = h * w
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > total_area * max_area_ratio:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            rw, rh = max(rw, rh), min(rw, rh)
            if rh < 2:
                continue
            aspect = rw / rh
            x1 = int(cx - rw / 2)
            y1 = int(cy - rh / 2)
            x2 = int(cx + rw / 2)
            y2 = int(cy + rh / 2)
            # Heuristic labels
            if aspect > 1.5:
                label = WINDOW_LABEL
            elif aspect < 1.2 and area < total_area * 0.02:
                label = DOOR_LABEL
            else:
                label = FIXTURE_LABEL
            results.append(
                DetectedObject(
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=0.8,
                )
            )
        return results

    def detect(
        self,
        image: np.ndarray,
        binary: np.ndarray = None,
        use_ml: bool = True,
    ) -> List[DetectedObject]:
        """
        Detect doors, windows, and fixtures.
        Returns DetectedObject list for legacy API; use detect_objects() for dict list.
        """
        raw = self.detect_objects(image)
        return [
            DetectedObject(
                label=d["class"],
                bbox=d["bbox"],
                confidence=d["confidence"],
            )
            for d in raw
        ]
