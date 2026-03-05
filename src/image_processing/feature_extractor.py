"""Extract structured features (rooms, connectivity) from blueprint and detections."""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from config import settings
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from src.image_processing.wall_detector import WallDetector
from src.image_processing.object_detector import ObjectDetector, DetectedObject


@dataclass
class RoomRegion:
    """A closed region (room) with optional label and area."""
    contour: np.ndarray
    area_px: float
    centroid: Tuple[float, float]
    label: str = "room"


@dataclass
class FloorPlanFeatures:
    """Aggregated features from a single floor plan image."""
    walls: List[Tuple[float, float, float, float]] = field(default_factory=list)
    doors: List[DetectedObject] = field(default_factory=list)
    windows: List[DetectedObject] = field(default_factory=list)
    fixtures: List[DetectedObject] = field(default_factory=list)
    rooms: List[RoomRegion] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)


class FeatureExtractor:
    """Extract walls, openings, and rooms from preprocessed image and detectors."""

    def __init__(
        self,
        wall_detector: WallDetector = None,
        object_detector: ObjectDetector = None,
        min_room_area: int = None,
    ):
        self.wall_detector = wall_detector or WallDetector()
        self.object_detector = object_detector or ObjectDetector()
        self.min_room_area = min_room_area or settings.MIN_ROOM_AREA

    def extract_rooms(self, binary: np.ndarray) -> List[RoomRegion]:
        """Extract room regions as contours (inverted: rooms = connected components of non-wall)."""
        # Walls are white in typical blueprint; rooms are enclosed regions
        # Invert so walls become 0, then find connected components of 1
        inv = 255 - binary
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv)
        rooms = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_room_area:
                continue
            cx, cy = centroids[i]
            # Contour from label mask
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            rooms.append(
                RoomRegion(
                    contour=cnt,
                    area_px=float(area),
                    centroid=(float(cx), float(cy)),
                    label="room",
                )
            )
        return rooms

    def extract(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        *,
        detect_rooms: bool = True,
    ) -> FloorPlanFeatures:
        """
        Run wall detection, object detection, and optional room extraction.
        Returns FloorPlanFeatures with walls, doors, windows, fixtures, rooms.
        """
        walls = self.wall_detector.detect(binary, merge=True)
        objects = self.object_detector.detect(image, binary=binary)
        doors = [o for o in objects if o.label == "door"]
        windows = [o for o in objects if o.label == "window"]
        fixtures = [o for o in objects if o.label == "fixture"]
        rooms = self.extract_rooms(binary) if detect_rooms else []
        h, w = binary.shape[:2]
        return FloorPlanFeatures(
            walls=walls,
            doors=doors,
            windows=windows,
            fixtures=fixtures,
            rooms=rooms,
            image_shape=(w, h),
        )
