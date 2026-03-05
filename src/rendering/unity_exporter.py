"""Export 3D scene and metadata for Unity import."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    import trimesh
except ImportError:
    trimesh = None


class UnityExporter:
    """Export mesh + scene description (lights, assets) for Unity."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_mesh_fbx(self, mesh: "trimesh.Trimesh", filepath: str) -> Optional[Path]:
        """Export mesh as FBX if supported; otherwise OBJ for Unity."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if trimesh is None:
            return None
        # Unity imports OBJ and FBX; trimesh exports OBJ/GLTF natively
        if path.suffix.lower() == ".fbx":
            mesh.export(str(path.with_suffix(".obj")))
            return path.with_suffix(".obj")
        mesh.export(str(path))
        return path

    def export_scene_json(
        self,
        scene: Dict[str, Any],
        name: str = "scene",
    ) -> Path:
        """Export scene description (lights, asset placements, mesh refs) as JSON."""
        path = self.output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2)
        return path

    def build_scene(
        self,
        mesh_paths: List[str],
        lights: List[Dict[str, Any]] = None,
        placements: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build scene dict for Unity loader: meshes, lights, object placements."""
        return {
            "version": 1,
            "meshes": [str(p) for p in mesh_paths],
            "lights": lights or [],
            "placements": placements or [],
        }

    @staticmethod
    def export_scene(mesh_data: Dict[str, Any], output_path: str) -> None:
        """
        Export complete scene structure for Unity (spec API).
        Creates scene_structure.json with walls, rooms, materials.
        """
        from pathlib import Path
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        scene_structure = {
            "version": "1.0",
            "objects": [],
            "materials": [],
            "textures": [],
        }
        walls = mesh_data.get("walls", mesh_data.get("edges", []))
        for i, wall in enumerate(walls):
            scene_structure["objects"].append({
                "id": f"wall_{i}",
                "type": "mesh",
                "geometry": wall,
                "material": "wall_material",
            })
        for i, room in enumerate(mesh_data.get("rooms", [])):
            scene_structure["objects"].append({
                "id": f"room_{i}",
                "type": "room",
                "geometry": room,
                "material": "floor_material",
            })
        out = Path(output_path) / "scene_structure.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(scene_structure, f, indent=2)
        log.info("Exported Unity scene to %s", out)
