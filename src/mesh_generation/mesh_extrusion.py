"""Extrude 2D floor plan to 3D mesh (walls with optional openings)."""

from typing import List, Optional, Tuple, Dict, Any
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

from src.mesh_generation.floor_plan_parser import ParsedFloorPlan, WallSegment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeshExtruder:
    """Extrude 2D wall segments to 3D boxes and apply door/window openings."""

    def __init__(
        self,
        wall_height: float = None,
        wall_thickness: float = None,
        door_height: float = None,
        window_height: float = None,
        pixel_to_meter: float = None,
    ):
        self.wall_height = wall_height or settings.DEFAULT_WALL_HEIGHT
        self.wall_thickness = wall_thickness or settings.DEFAULT_WALL_THICKNESS
        self.door_height = door_height or settings.DOOR_HEIGHT
        self.window_height = window_height or settings.WINDOW_HEIGHT
        self.pixel_to_meter = pixel_to_meter if pixel_to_meter is not None else getattr(settings, "PIXEL_TO_METER", 0.01)
        self.mesh: Optional["trimesh.Trimesh"] = None

    def _wall_quad_2d(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        thickness: float,
    ) -> np.ndarray:
        """Four corners of wall strip in XY (counterclockwise from bottom-left)."""
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        if L < 1e-9:
            return np.array([]).reshape(0, 2)
        nx_ = -dy / L
        ny_ = dx / L
        half = thickness / 2
        p1 = (x1 + nx_ * half, y1 + ny_ * half)
        p2 = (x2 + nx_ * half, y2 + ny_ * half)
        p3 = (x2 - nx_ * half, y2 - ny_ * half)
        p4 = (x1 - nx_ * half, y1 - ny_ * half)
        return np.array([p1, p2, p3, p4], dtype=np.float64)

    def _extrude_quad_to_box(
        self,
        quad_xy: np.ndarray,
        z_low: float = 0.0,
        z_high: float = None,
    ) -> Optional["trimesh.Trimesh"]:
        """Extrude a quad (4 vertices in XY) to a box (8 vertices, 12 triangles)."""
        if trimesh is None:
            return None
        z_high = z_high or self.wall_height
        verts = []
        for (x, y) in quad_xy:
            verts.append([x, y, z_low])
        for (x, y) in quad_xy:
            verts.append([x, y, z_high])
        verts = np.array(verts, dtype=np.float64)
        # Tri mesh for a box: 6 faces, 2 tris each
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 1], [1, 4, 5],  # front
            [1, 5, 2], [2, 5, 6],
            [2, 6, 3], [3, 6, 7],
            [3, 7, 0], [0, 7, 4],
        ]
        return trimesh.Trimesh(vertices=verts, faces=faces)

    def _wall_mesh_simple(self, seg: WallSegment) -> Optional["trimesh.Trimesh"]:
        """Single wall segment as solid box (no openings)."""
        quad = self._wall_quad_2d(
            seg.x1_m, seg.y1_m,
            seg.x2_m, seg.y2_m,
            self.wall_thickness,
        )
        if len(quad) < 4:
            return None
        return self._extrude_quad_to_box(quad)

    def extrude_floor_plan(
        self,
        plan: ParsedFloorPlan,
        combine: bool = True,
    ) -> Optional["trimesh.Trimesh"]:
        """
        Extrude all walls to 3D. If combine=True, merge into single mesh.
        Openings can be applied via CSG subtract in a separate step.
        """
        if trimesh is None:
            return None
        meshes = []
        for wall in plan.walls:
            m = self._wall_mesh_simple(wall)
            if m is not None:
                meshes.append(m)
        if not meshes:
            return None
        if combine:
            return trimesh.util.concatenate(meshes)
        return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

    # ---- Spec pipeline: graph_data-based API ----

    def extrude_walls(self, graph_data: Dict[str, Any]) -> Optional["trimesh.Trimesh"]:
        """Extrude walls from 2D graph (edges with x1,y1,x2,y2). Coordinates scaled to meters."""
        if trimesh is None:
            return None
        s = self.pixel_to_meter
        meshes = []
        for edge in graph_data.get("edges", []):
            x1, y1 = edge["x1"] * s, edge["y1"] * s
            x2, y2 = edge["x2"] * s, edge["y2"] * s
            wall_mesh = self._create_wall_box(
                (x1, y1), (x2, y2),
                self.wall_height,
                self.wall_thickness,
            )
            if wall_mesh is not None:
                meshes.append(wall_mesh)
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            logger.info("Extruded %d walls", len(meshes))
            return combined
        return None

    def _create_wall_box(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        height: float,
        thickness: float,
    ) -> Optional["trimesh.Trimesh"]:
        """Create a rectangular box representing a wall segment."""
        if trimesh is None:
            return None
        x1, y1 = start
        x2, y2 = end
        wall_vec = np.array([x2 - x1, y2 - y1, 0.0])
        wall_length = np.linalg.norm(wall_vec[:2])
        if wall_length < 1e-6:
            return None
        wall_vec = wall_vec / wall_length
        perp = np.array([-wall_vec[1], wall_vec[0], 0.0])
        p1 = np.array([x1, y1, 0.0]) - perp * thickness / 2
        p2 = np.array([x1, y1, 0.0]) + perp * thickness / 2
        p3 = np.array([x2, y2, 0.0]) + perp * thickness / 2
        p4 = np.array([x2, y2, 0.0]) - perp * thickness / 2
        p5 = p1 + np.array([0, 0, height])
        p6 = p2 + np.array([0, 0, height])
        p7 = p3 + np.array([0, 0, height])
        p8 = p4 + np.array([0, 0, height])
        vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def create_floors_and_ceilings(
        self, rooms: List[Dict[str, Any]]
    ) -> List["trimesh.Trimesh"]:
        """Create floor and ceiling meshes for each room. Polygon XY scaled to meters."""
        if trimesh is None:
            return []
        s = self.pixel_to_meter
        meshes = []
        for room in rooms:
            polygon = room.get("polygon")
            if polygon is None:
                continue
            polygon = np.array(polygon, dtype=np.float64) * s
            if len(polygon) < 3:
                continue
            floor_vertices = np.column_stack([polygon, np.zeros(len(polygon))])
            floor_faces = self._triangulate_polygon(polygon)
            if floor_faces is not None and len(floor_faces) > 0:
                floor = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
                meshes.append(floor)
            ceiling_vertices = np.column_stack([
                polygon,
                np.full(len(polygon), self.wall_height),
            ])
            if floor_faces is not None and len(floor_faces) > 0:
                ceiling = trimesh.Trimesh(vertices=ceiling_vertices, faces=floor_faces)
                meshes.append(ceiling)
        logger.info("Created %d floor/ceiling meshes", len(rooms) * 2)
        return meshes

    @staticmethod
    def _triangulate_polygon(polygon: np.ndarray) -> Optional[np.ndarray]:
        """Triangulate polygon using fan triangulation (convex or simple)."""
        if len(polygon) < 3:
            return None
        faces = []
        for i in range(1, len(polygon) - 1):
            faces.append([0, i, i + 1])
        return np.array(faces)

    def create_complete_mesh(self, graph_data: Dict[str, Any]) -> Optional["trimesh.Trimesh"]:
        """Create complete 3D mesh from 2D floor plan data (XY in meters, centered at origin)."""
        if trimesh is None:
            return None
        all_meshes = []
        wall_mesh = self.extrude_walls(graph_data)
        if wall_mesh is not None:
            all_meshes.append(wall_mesh)
        rooms = graph_data.get("rooms", [])
        floor_meshes = self.create_floors_and_ceilings(rooms)
        all_meshes.extend(floor_meshes)
        if all_meshes:
            combined = trimesh.util.concatenate(all_meshes)
            # Center XY at origin so viewers show it properly; keep Z (floor at 0)
            try:
                center = combined.centroid
                combined.vertices[:, 0] -= center[0]
                combined.vertices[:, 1] -= center[1]
            except Exception:
                pass
            self.mesh = combined
            logger.info("Created complete mesh with %d vertices", combined.vertices.shape[0])
            return combined
        return None

    def export_mesh(self, output_path: str) -> None:
        """Export mesh to file (OBJ, STL, GLB, etc.)."""
        if self.mesh is not None:
            self.mesh.export(output_path)
            logger.info("Exported mesh to %s", output_path)
        else:
            logger.warning("No mesh to export")
