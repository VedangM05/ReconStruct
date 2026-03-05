"""Blueprint image preprocessing: normalize, denoise, binarize, deskew."""

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from config import settings
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlueprintPreprocessor:
    """Preprocess blueprint images for wall/object detection."""

    def __init__(
        self,
        target_resolution: Tuple[int, int] = None,
        target_dpi: int = None,
        denoise_strength: int = 10,
    ):
        self.target_resolution = target_resolution or settings.TARGET_RESOLUTION
        self.target_dpi = target_dpi or settings.TARGET_DPI
        self.denoise_strength = denoise_strength

    def load(self, path: str) -> np.ndarray:
        """Load image from path; return BGR array."""
        return self.load_image(path)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load blueprint image with validation (spec API)."""
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.imread(str(path))
        if img is None:
            img = np.array(Image.open(path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        logger.info("Loaded image: %s", img.shape)
        return img

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target resolution (width, height)."""
        return self.standardize_resolution(image)

    def standardize_resolution(self, image: np.ndarray) -> np.ndarray:
        """Resize image to standard resolution (spec API)."""
        resized = cv2.resize(
            image, self.target_resolution, interpolation=cv2.INTER_CUBIC
        )
        logger.info("Resized to %s", self.target_resolution)
        return resized

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        return self.convert_to_grayscale(image)

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale for line detection (spec API)."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise while preserving edges (bilateral for grayscale spec)."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, self.denoise_strength, self.denoise_strength, 7, 21
            )
        # Spec: bilateral filtering for edge-preserving denoise
        return cv2.bilateralFilter(image, 9, 75, 75)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Binarize for line/structure detection (walls as white on black)."""
        if len(gray.shape) == 3:
            gray = self.convert_to_grayscale(gray)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        logger.info("Applied Otsu's thresholding")
        return binary

    def normalize_orientation(self, image: np.ndarray) -> np.ndarray:
        """Optional: deskew or align if EXIF/orientation metadata present."""
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (spec API)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(
            image if len(image.shape) == 2 else self.convert_to_grayscale(image)
        )
        logger.info("Applied CLAHE contrast enhancement")
        return enhanced

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image skew using Hough Transform. Only corrects small skew (0.5°–15°) to avoid breaking axis-aligned blueprints."""
        gray = (
            self.convert_to_grayscale(image)
            if len(image.shape) == 3
            else image
        )
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is None or len(lines) == 0:
            return image
        angles = [
            np.abs(90 - np.degrees(line[0][1])) for line in lines
        ]
        angle = float(np.median(angles))
        if angle > 45:
            angle = 90 - angle
        # Only rotate for small skew; skip if nearly 0 (already straight) or large (noise, would break H/V lines)
        if abs(angle) < 0.5:
            return image
        if abs(angle) > 15:
            logger.info("Skipping deskew (angle %.2f° likely noise)", angle)
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
        )
        logger.info("Deskewed by %.2f degrees", angle)
        return rotated

    def run(
        self,
        path: str,
        *,
        denoise: bool = True,
        binarize_output: bool = False,
    ) -> dict:
        """
        Run full preprocessing pipeline.
        Returns dict with keys: image (BGR), gray, binary (if binarize_output), shape.
        """
        image = self.load(path)
        image = self.normalize_orientation(image)
        image = self.resize(image)
        if denoise:
            image = self.denoise(image)
        gray = self.to_grayscale(image)
        out = {
            "image": image,
            "gray": gray,
            "shape": gray.shape,
        }
        if binarize_output:
            out["binary"] = self.binarize(gray)
        return out

    def preprocess_full_pipeline(self, image_path: str) -> tuple:
        """
        Full preprocessing pipeline: load → standardize → denoise → enhance → deskew → binarize.
        Returns: (original, grayscale, binary, preprocessed)
        """
        logger.info("Starting preprocessing pipeline: %s", image_path)
        image = self.load_image(image_path)
        image = self.standardize_resolution(image)
        gray = self.convert_to_grayscale(image)
        gray = self.denoise(gray)
        gray = self.enhance_contrast(gray)
        gray = self.deskew(gray)
        binary = self.binarize(gray)
        logger.info("Preprocessing complete")
        return image, gray, binary, gray

    def save_intermediate_results(
        self,
        output_dir: str,
        prefix: str,
        original: np.ndarray,
        gray: np.ndarray,
        binary: np.ndarray,
    ) -> None:
        """Save preprocessing stages for debugging."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{prefix}_01_original.png", original)
        cv2.imwrite(f"{output_dir}/{prefix}_02_grayscale.png", gray)
        cv2.imwrite(f"{output_dir}/{prefix}_03_binary.png", binary)
        logger.info("Saved intermediate results to %s", output_dir)
