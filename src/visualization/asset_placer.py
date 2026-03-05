"""Place 3D assets (furniture, fixtures) in rooms from floor plan data."""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from config.paths import ASSETS_3D, ASSETS_DIR
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config.paths import ASSETS_3D, ASSETS_DIR

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AssetPlacer:
    """Load and place 3D asset instances in the scene based on room regions and rules."""

    def __init__(self, assets_dir: str = None):
        self.assets_dir = Path(assets_dir) if assets_dir else ASSETS_3D
        self._cache = {}
        self.asset_library = self._load_asset_library()

    def _load_asset_library(self) -> Dict[str, Any]:
        """Load available 3D assets from library (spec API)."""
        library = {
            "door": self._create_default_door(),
            "window_frame": self._create_default_window_frame(),
            "furniture": {},
        }
        return library

    def _create_default_door(self) -> Optional["trimesh.Trimesh"]:
        """Create default door geometry (spec API)."""
        if trimesh is None:
            return None
        vertices = np.array([
            [0, 0, 0], [0.9, 0, 0],
            [0.9, 0.05, 0], [0, 0.05, 0],
            [0, 0, 2.1], [0.9, 0, 2.1],
            [0.9, 0.05, 2.1], [0, 0.05, 2.1],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _create_default_window_frame(self) -> Optional["trimesh.Trimesh"]:
        """Create default window frame geometry (spec API)."""
        if trimesh is None:
            return None
        vertices = np.array([
            [0, -0.05, 1.0], [1.5, -0.05, 1.0],
            [1.5, 0.05, 1.0], [0, 0.05, 1.0],
            [0, -0.05, 2.2], [1.5, -0.05, 2.2],
            [1.5, 0.05, 2.2], [0, 0.05, 2.2],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def place_doors(
        self,
        scene_mesh: "trimesh.Trimesh",
        doors: List[Dict[str, Any]],
    ) -> Optional["trimesh.Trimesh"]:
        """Place doors in scene at detected locations (spec API)."""
        if trimesh is None:
            return scene_mesh
        meshes = [scene_mesh]
        for door in doors:
            loc = door.get("location")
            if loc is None:
                continue
            x, y = loc[0], loc[1]
            door_mesh = self._create_default_door()
            if door_mesh is not None:
                door_mesh.apply_translation([x, y, 0])
                meshes.append(door_mesh)
        result = trimesh.util.concatenate(meshes)
        logger.info("Placed %d doors", len(doors))
        return result

    def place_windows(
        self,
        scene_mesh: "trimesh.Trimesh",
        windows: List[Dict[str, Any]],
    ) -> Optional["trimesh.Trimesh"]:
        """Place windows in scene at detected locations (spec API)."""
        if trimesh is None:
            return scene_mesh
        meshes = [scene_mesh]
        for window in windows:
            loc = window.get("location")
            if loc is None:
                continue
            x, y = loc[0], loc[1]
            window_mesh = self._create_default_window_frame()
            if window_mesh is not None:
                window_mesh.apply_translation([x, y, 0])
                meshes.append(window_mesh)
        result = trimesh.util.concatenate(meshes)
        logger.info("Placed %d windows", len(windows))
        return result

    def list_assets(self, category: str = None) -> List[str]:
        """List available asset filenames (e.g. by category subfolder)."""
        base = self.assets_dir / category if category else self.assets_dir
        if not base.exists():
            return []
        return [f.name for f in base.iterdir() if f.suffix.lower() in (".obj", ".glb", ".gltf")]

    def load_asset(self, relative_path: str) -> Optional["trimesh.Trimesh"]:
        """Load a single asset mesh from assets_dir."""
        if trimesh is None:
            return None
        path = self.assets_dir / relative_path
        if not path.exists():
            return None
        if path.suffix.lower() in (".obj", ".glb", ".gltf"):
            return trimesh.load(str(path), force="mesh")
        return None

    def place_in_room(
        self,
        room_centroid_xy: Tuple[float, float],
        room_area_m2: float,
        asset_id: str,
        transform: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Return placement record: asset_id, position [x,y,z], scale, rotation for export."""
        transform = transform or {}
        return {
            "asset_id": asset_id,
            "position": [
                room_centroid_xy[0],
                room_centroid_xy[1],
                transform.get("z", 0.0),
            ],
            "scale": transform.get("scale", [1.0, 1.0, 1.0]),
            "rotation_deg": transform.get("rotation_deg", 0.0),
        }

    def suggest_placements(
        self,
        rooms: List[Dict[str, Any]],
        rules: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest asset placements per room based on area/type rules. Returns list of place_in_room records."""
        rules = rules or {}
        placements = []
        for room in rooms:
            centroid = room.get("centroid", (0, 0))
            area = room.get("area_m2", 0)
            for asset_id, rule in rules.items():
                if rule.get("min_area", 0) <= area:
                    placements.append(
                        self.place_in_room(centroid, area, asset_id, rule.get("transform", {}))
                    )
        return placements
