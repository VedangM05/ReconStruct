"""Wall detection from blueprint images (line detection + optional ML)."""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from config import settings
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WallDetector:
    """Detect wall lines from preprocessed blueprint images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_wall_length: int = None,
        merge_threshold_px: float = 5.0,
        angle_tolerance_deg: float = 5.0,
    ):
        self.model_path = model_path or settings.WALL_DETECTION_MODEL
        self.min_wall_length = min_wall_length or settings.MIN_WALL_LENGTH
        self.merge_threshold_px = merge_threshold_px
        self.angle_tolerance_deg = angle_tolerance_deg
        self._model = None
        self.wall_segments: List[Tuple[float, float, float, float]] = []

    def _load_model(self):
        """Lazy load Keras/TF model if available."""
        if self._model is not None:
            return
        path = Path(self.model_path)
        if not path.exists():
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(path))
        except Exception:
            self._model = None

    def detect_lines_hough(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Detect lines using Probabilistic Hough Line Transform.
        Returns: array of lines with format [[x1, y1, x2, y2], ...]
        """
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=self.min_wall_length,
            maxLineGap=10,
        )
        if lines is None:
            logger.warning("No lines detected")
            return np.array([]).reshape(0, 4)
        lines = lines.squeeze()
        if lines.ndim == 1:
            lines = lines.reshape(1, -1)
        logger.info("Detected %d initial line segments", len(lines))
        return lines

    def extract_line_features(self, lines: np.ndarray) -> np.ndarray:
        """
        Extract features from each line for classification.
        Features: length, angle, straightness, confidence
        """
        if len(lines) == 0:
            return np.array([]).reshape(0, 4)
        features = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            straightness = 1.0
            confidence = min(1.0, length / 500)
            features.append([length, angle, straightness, confidence])
        features = np.array(features)
        logger.info("Extracted features for %d lines", len(features))
        return features

    def filter_walls_logistic_regression(
        self, lines: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """
        Classify lines as walls or non-walls (heuristic: long, horizontal/vertical).
        """
        if len(features) == 0:
            return np.array([])
        min_length = 50
        angle_tolerance = 15
        is_wall = []
        rejected_length = 0
        rejected_angle = 0
        for feature in features:
            length, angle, straightness, confidence = feature
            if length < min_length:
                is_wall.append(False)
                rejected_length += 1
                continue
            normalized_angle = min(angle, 180 - angle)
            if normalized_angle < angle_tolerance or normalized_angle > (
                90 - angle_tolerance
            ):
                is_wall.append(True)
            else:
                is_wall.append(False)
                rejected_angle += 1
        is_wall = np.array(is_wall)
        wall_lines = lines[is_wall]
        logger.info(
            "Classified %d lines as walls (rejected: %d by length < %d, %d by angle not H/V)",
            np.sum(is_wall), rejected_length, min_length, rejected_angle,
        )
        return wall_lines

    def merge_collinear_segments(
        self, lines: np.ndarray, tolerance: float = 10
    ) -> np.ndarray:
        """
        Merge collinear wall segments that are close together.
        """
        if len(lines) == 0:
            return np.array([])
        merged = []
        used = set()
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            x1, y1, x2, y2 = line1[:4]
            merged_line = [float(x1), float(y1), float(x2), float(y2)]
            used.add(i)
            for j, line2 in enumerate(lines):
                if j in used or j <= i:
                    continue
                x3, y3, x4, y4 = line2[:4]
                if self._are_collinear(
                    (x1, y1, x2, y2), (x3, y3, x4, y4), tolerance
                ):
                    all_x = [x1, x2, x3, x4]
                    all_y = [y1, y2, y3, y4]
                    merged_line = [
                        min(all_x), min(all_y), max(all_x), max(all_y)
                    ]
                    used.add(j)
            merged.append(merged_line)
        merged_array = np.array(merged)
        logger.info("Merged to %d wall segments", len(merged_array))
        return merged_array

    @staticmethod
    def _are_collinear(line1, line2, tolerance: float) -> bool:
        """Check if two lines are collinear within tolerance."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        def point_to_line_distance(px, py, x1, y1, x2, y2):
            numerator = abs(
                (y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1
            )
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            return numerator / (denominator + 1e-6)

        d1 = point_to_line_distance(x3, y3, x1, y1, x2, y2)
        d2 = point_to_line_distance(x4, y4, x1, y1, x2, y2)
        return max(d1, d2) < tolerance

    def detect_walls(
        self, binary_image: np.ndarray, debug_output_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        Complete wall detection pipeline: Hough → features → classification → merging.
        If debug_output_dir is set, saves stage images and logs angle distribution.
        """
        lines = self.detect_lines_hough(binary_image)
        if len(lines) == 0:
            self.wall_segments = []
            if debug_output_dir:
                Path(debug_output_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{debug_output_dir}/debug_walls_0_binary.png", binary_image)
                logger.info("Debug: saved debug_walls_0_binary.png (no Hough lines)")
            return np.array([])

        features = self.extract_line_features(lines)
        wall_lines = self.filter_walls_logistic_regression(lines, features)
        wall_segments = self.merge_collinear_segments(wall_lines)
        self.wall_segments = wall_segments

        if debug_output_dir:
            self._save_wall_debug(
                binary_image, lines, wall_lines, wall_segments, debug_output_dir
            )
        return wall_segments

    def _save_wall_debug(
        self,
        binary_image: np.ndarray,
        all_lines: np.ndarray,
        wall_lines: np.ndarray,
        wall_segments: np.ndarray,
        output_dir: str,
    ) -> None:
        """Save debug images and log angle distribution for wall detection."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = np.zeros_like(binary_image, dtype=np.uint8)
        if len(base.shape) == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        h, w = base.shape[:2]

        # 0: input binary
        cv2.imwrite(str(out / "debug_walls_0_binary.png"), binary_image)

        # 1: all Hough lines (red)
        viz1 = base.copy()
        for line in all_lines:
            x1, y1, x2, y2 = map(int, line[:4])
            cv2.line(viz1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(out / "debug_walls_1_hough_all.png"), viz1)
        logger.info("Debug: saved debug_walls_1_hough_all.png (%d lines)", len(all_lines))

        # 2: after wall filter, before merge (green)
        viz2 = base.copy()
        for line in wall_lines:
            x1, y1, x2, y2 = map(int, line[:4])
            cv2.line(viz2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out / "debug_walls_2_after_filter.png"), viz2)
        logger.info("Debug: saved debug_walls_2_after_filter.png (%d lines)", len(wall_lines))

        # 3: final merged segments (blue)
        viz3 = base.copy()
        for seg in wall_segments:
            x1, y1, x2, y2 = map(int, seg[:4])
            cv2.line(viz3, (x1, y1), (x2, y2), (255, 128, 0), 3)
        cv2.imwrite(str(out / "debug_walls_3_merged.png"), viz3)
        logger.info("Debug: saved debug_walls_3_merged.png (%d segments)", len(wall_segments))

        # Angle distribution of all Hough lines (for tuning angle_tolerance)
        if len(all_lines) > 0:
            angles_deg = np.degrees(np.arctan2(
                all_lines[:, 3] - all_lines[:, 1],
                all_lines[:, 2] - all_lines[:, 0],
            )) % 180
            # Bucket 0-15 (H), 75-90 (V), 15-75 (diagonal)
            h_count = np.sum((angles_deg < 15) | (angles_deg > 165))
            v_count = np.sum((angles_deg >= 75) & (angles_deg <= 105))
            diag_count = len(angles_deg) - h_count - v_count
            logger.info(
                "Debug: angle distribution (all Hough) — horizontal-ish: %d, vertical-ish: %d, diagonal: %d",
                h_count, v_count, diag_count,
            )

    def visualize_walls(
        self, image: np.ndarray, walls: np.ndarray
    ) -> np.ndarray:
        """Visualize detected walls on image."""
        viz = image.copy()
        if len(image.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
        for wall in walls:
            x1, y1, x2, y2 = map(int, wall[:4])
            cv2.line(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return viz

    def _segment_to_line(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
        """Convert segment to (angle_rad, dist, length)."""
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return 0.0, 0.0, 0.0
        angle = np.arctan2(dy, dx)
        # Distance from origin to line
        dist = abs(x1 * dy - y1 * dx) / length
        return angle, dist, length

    def merge_collinear(
        self,
        segments: np.ndarray,
        angle_tol: float = None,
        dist_tol: float = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Merge nearly collinear segments into longer walls."""
        angle_tol = angle_tol or np.radians(self.angle_tolerance_deg)
        dist_tol = dist_tol or self.merge_threshold_px
        if len(segments) == 0:
            return []
        # Group by similar angle and proximity, then merge endpoints
        merged = []
        used = [False] * len(segments)
        for i, (x1, y1, x2, y2) in enumerate(segments):
            if used[i]:
                continue
            group = [(x1, y1, x2, y2)]
            used[i] = True
            ang, dist, length = self._segment_to_line(x1, y1, x2, y2)
            for j, (a1, b1, a2, b2) in enumerate(segments):
                if used[j]:
                    continue
                ang2, dist2, _ = self._segment_to_line(a1, b1, a2, b2)
                if abs(ang - ang2) <= angle_tol and abs(dist - dist2) <= dist_tol:
                    group.append((a1, b1, a2, b2))
                    used[j] = True
            # Merge group into one segment (bounding line)
            xs = [p[0] for s in group for p in [(s[0], s[1]), (s[2], s[3])]]
            ys = [p[1] for s in group for p in [(s[0], s[1]), (s[2], s[3])]]
            # Project endpoints along dominant angle
            cx, cy = np.mean(xs), np.mean(ys)
            u = np.cos(ang)
            v = np.sin(ang)
            projs = [(x - cx) * u + (y - cy) * v for x, y in zip(xs, ys)]
            lo, hi = min(projs), max(projs)
            x1_ = cx + lo * u
            y1_ = cy + lo * v
            x2_ = cx + hi * u
            y2_ = cy + hi * v
            merged.append((float(x1_), float(y1_), float(x2_), float(y2_)))
        return merged

    def detect(self, binary: np.ndarray, merge: bool = True) -> List[Tuple[float, float, float, float]]:
        """
        Detect wall segments from binary blueprint image.
        Returns list of (x1, y1, x2, y2) in pixel coordinates.
        """
        segments = self.detect_walls(binary)
        if len(segments) == 0:
            return []
        return [tuple(s) for s in segments]
