"""CSG (Constructive Solid Geometry) for subtracting openings from walls."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from config import settings
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from src.mesh_generation.floor_plan_parser import WallSegment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSGOperations:
    """Boolean operations on meshes (subtract openings from wall volumes)."""

    def __init__(
        self,
        wall_height: float = None,
        door_height: float = None,
        window_height: float = None,
    ):
        self.wall_height = wall_height or settings.DEFAULT_WALL_HEIGHT
        self.door_height = door_height or settings.DOOR_HEIGHT
        self.window_height = window_height or settings.WINDOW_HEIGHT

    def _opening_box(
        self,
        seg: WallSegment,
        t: float,
        width_m: float,
        height_m: float,
        opening_type: str,
    ) -> Optional["trimesh.Trimesh"]:
        """Create a box mesh for a door/window opening along a wall segment."""
        if trimesh is None:
            return None
        x = seg.x1_m + t * (seg.x2_m - seg.x1_m)
        y = seg.y1_m + t * (seg.y2_m - seg.y1_m)
        dx = seg.x2_m - seg.x1_m
        dy = seg.y2_m - seg.y1_m
        L = np.hypot(dx, dy)
        if L < 1e-9:
            return None
        nx_ = -dy / L
        ny_ = dx / L
        # Center of opening at (x,y), half-width along wall, full thickness through wall
        half_w = width_m / 2
        thick = settings.DEFAULT_WALL_THICKNESS * 1.1  # slight overlap for clean boolean
        # Box corners in XY (local): along wall ± half_w, perpendicular ± thick/2
        corners = [
            (x - (dx / L) * half_w + nx_ * thick / 2, y - (dy / L) * half_w + ny_ * thick / 2),
            (x + (dx / L) * half_w + nx_ * thick / 2, y + (dy / L) * half_w + ny_ * thick / 2),
            (x + (dx / L) * half_w - nx_ * thick / 2, y + (dy / L) * half_w - ny_ * thick / 2),
            (x - (dx / L) * half_w - nx_ * thick / 2, y - (dy / L) * half_w - ny_ * thick / 2),
        ]
        z_bottom = 0.0
        z_top = height_m
        verts = []
        for (px, py) in corners:
            verts.append([px, py, z_bottom])
        for (px, py) in corners:
            verts.append([px, py, z_top])
        verts = np.array(verts, dtype=np.float64)
        faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 6], [2, 6, 3], [3, 6, 7], [3, 7, 0], [0, 7, 4],
        ]
        return trimesh.Trimesh(vertices=verts, faces=faces)

    def subtract_openings(
        self,
        wall_mesh: "trimesh.Trimesh",
        segment: WallSegment,
    ) -> Optional["trimesh.Trimesh"]:
        """
        Subtract door/window boxes from a wall mesh. Uses trimesh boolean_diff if available.
        """
        if trimesh is None or not segment.openings:
            return wall_mesh
        result = wall_mesh
        for opening in segment.openings:
            otype = opening.get("type", "door")
            height = self.door_height if otype == "door" else self.window_height
            box = self._opening_box(
                segment,
                opening.get("t", 0.5),
                opening.get("width_m", 0.9),
                height,
                otype,
            )
            if box is None:
                continue
            try:
                result = result.difference(box, engine="blender")
            except Exception:
                try:
                    result = result.difference(box, engine="scad")
                except Exception:
                    pass  # keep mesh without opening
        return result

    @staticmethod
    def create_door_opening(
        wall_mesh: "trimesh.Trimesh",
        door_location: Tuple[float, float],
        door_width: float = None,
        door_height: float = None,
    ) -> Optional["trimesh.Trimesh"]:
        """
        Create a door opening by subtracting a box from wall mesh.
        door_location is (x, y) center in same coordinate system as wall.
        """
        if trimesh is None:
            return None
        door_width = door_width or settings.DOOR_WIDTH
        door_height = door_height or settings.DOOR_HEIGHT
        x, y = door_location
        door_box_vertices = np.array([
            [x - door_width / 2, y - 0.1, 0],
            [x + door_width / 2, y - 0.1, 0],
            [x + door_width / 2, y + 0.1, 0],
            [x - door_width / 2, y + 0.1, 0],
            [x - door_width / 2, y - 0.1, door_height],
            [x + door_width / 2, y - 0.1, door_height],
            [x + door_width / 2, y + 0.1, door_height],
            [x - door_width / 2, y + 0.1, door_height],
        ], dtype=np.float64)
        door_box_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        door_box = trimesh.Trimesh(vertices=door_box_vertices, faces=door_box_faces)
        try:
            result = trimesh.boolean.difference(wall_mesh, door_box)
            logger.info("Created door opening at (%.2f, %.2f)", x, y)
            return result
        except Exception:
            return wall_mesh

    @staticmethod
    def create_window_opening(
        wall_mesh: "trimesh.Trimesh",
        window_location: Tuple[float, float],
        window_width: float = None,
        window_height: float = None,
        sill_height: float = 1.0,
    ) -> Optional["trimesh.Trimesh"]:
        """Create a window opening at specified height in wall."""
        if trimesh is None:
            return None
        window_width = window_width or settings.WINDOW_WIDTH
        window_height = window_height or settings.WINDOW_HEIGHT
        x, y = window_location
        window_box_vertices = np.array([
            [x - window_width / 2, y - 0.1, sill_height],
            [x + window_width / 2, y - 0.1, sill_height],
            [x + window_width / 2, y + 0.1, sill_height],
            [x - window_width / 2, y + 0.1, sill_height],
            [x - window_width / 2, y - 0.1, sill_height + window_height],
            [x + window_width / 2, y - 0.1, sill_height + window_height],
            [x + window_width / 2, y + 0.1, sill_height + window_height],
            [x - window_width / 2, y + 0.1, sill_height + window_height],
        ], dtype=np.float64)
        window_box_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        window_box = trimesh.Trimesh(vertices=window_box_vertices, faces=window_box_faces)
        try:
            result = trimesh.boolean.difference(wall_mesh, window_box)
            logger.info("Created window opening at (%.2f, %.2f, %.2f)", x, y, sill_height)
            return result
        except Exception:
            return wall_mesh
