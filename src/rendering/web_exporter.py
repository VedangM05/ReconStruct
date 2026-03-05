"""Export 3D scene for web (GLTF/GLB, Three.js compatible)."""

from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import trimesh
except ImportError:
    trimesh = None


class WebExporter:
    """Export mesh and scene for web viewers (GLB/GLTF)."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_glb(self, mesh: "trimesh.Trimesh", filepath: str = None) -> Optional[Path]:
        """Export single mesh as GLB for web/Three.js."""
        if trimesh is None:
            return None
        path = Path(filepath) if filepath else self.output_dir / "scene.glb"
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="glb")
        return path

    def export_gltf(self, mesh: "trimesh.Trimesh", filepath: str = None) -> Optional[Path]:
        """Export as GLTF (JSON + bin) for web."""
        if trimesh is None:
            return None
        path = Path(filepath) if filepath else self.output_dir / "scene.gltf"
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="gltf")
        return path

    def export_for_three(
        self,
        mesh: "trimesh.Trimesh",
        base_name: str = "model",
    ) -> Dict[str, Any]:
        """Export GLB and return manifest for Three.js loader."""
        glb_path = self.output_dir / f"{base_name}.glb"
        self.export_glb(mesh, str(glb_path))
        return {
            "model_url": glb_path.name,
            "format": "glb",
        }
