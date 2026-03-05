"""Export 3D mesh and scene to Blender-compatible formats (OBJ, FBX, GLTF)."""

from pathlib import Path
from typing import Optional, List

try:
    import trimesh
except ImportError:
    trimesh = None

from config.constants import EXPORT_OBJ, EXPORT_FBX, EXPORT_GLTF, EXPORT_GLB


class BlenderExporter:
    """Export trimesh or scene description to file formats usable in Blender."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_mesh(
        self,
        mesh: "trimesh.Trimesh",
        filepath: str,
        format: str = EXPORT_OBJ,
    ) -> Path:
        """Export a single mesh to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if trimesh is None:
            raise RuntimeError("trimesh is required for mesh export")
        if format.lower() in (EXPORT_OBJ, ".obj"):
            mesh.export(str(path))
        elif format.lower() in (EXPORT_GLTF, ".gltf", EXPORT_GLB, ".glb"):
            mesh.export(str(path), file_type="gltf" if format == EXPORT_GLTF else "glb")
        else:
            mesh.export(str(path))
        return path

    def export_scene(
        self,
        meshes: List["trimesh.Trimesh"],
        base_name: str = "floor_plan",
        formats: List[str] = None,
    ) -> List[Path]:
        """Export multiple meshes (e.g. walls, floors) as a single combined mesh to each format."""
        formats = formats or [EXPORT_OBJ, EXPORT_GLB]
        if not meshes:
            return []
        combined = trimesh.util.concatenate(meshes) if trimesh else None
        if combined is None:
            return []
        paths = []
        for fmt in formats:
            ext = ".obj" if fmt == EXPORT_OBJ else ".glb" if fmt == EXPORT_GLB else ".gltf"
            out = self.output_dir / f"{base_name}{ext}"
            self.export_mesh(combined, str(out), format=fmt)
            paths.append(out)
        return paths
