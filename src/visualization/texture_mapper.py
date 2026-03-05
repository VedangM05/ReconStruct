"""Apply textures and materials to mesh geometry."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from config.paths import ASSETS_TEXTURES, ASSETS_MATERIALS
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config.paths import ASSETS_TEXTURES, ASSETS_MATERIALS


class TextureMapper:
    """Map textures and materials to wall/floor/ceiling meshes."""

    def __init__(
        self,
        textures_dir: Path = None,
        materials_dir: Path = None,
        quality: str = "high",
    ):
        self.textures_dir = Path(textures_dir) if textures_dir else ASSETS_TEXTURES
        self.materials_dir = Path(materials_dir) if materials_dir else ASSETS_MATERIALS
        self.quality = quality

    def list_textures(self) -> List[str]:
        """List available texture filenames."""
        if not self.textures_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        return [f.name for f in self.textures_dir.iterdir() if f.suffix.lower() in exts]

    def get_material_for_surface(self, surface_type: str) -> Dict[str, Any]:
        """Return material descriptor (name, diffuse_map path, uv_scale) for surface type."""
        defaults = {
            "wall": {"name": "wall", "uv_scale": [1.0, 1.0], "diffuse": None},
            "floor": {"name": "floor", "uv_scale": [1.0, 1.0], "diffuse": None},
            "ceiling": {"name": "ceiling", "uv_scale": [1.0, 1.0], "diffuse": None},
        }
        out = defaults.get(surface_type, defaults["wall"]).copy()
        # Try to resolve a texture file
        for name in ["wall", "floor", "ceiling"]:
            for ext in [".png", ".jpg"]:
                p = self.textures_dir / f"{name}{ext}"
                if p.exists():
                    if surface_type == name:
                        out["diffuse"] = str(p)
                    break
        return out

    def apply_uv_to_mesh(self, mesh: "trimesh.Trimesh", surface_type: str = "wall") -> "trimesh.Trimesh":
        """Assign UV coordinates to mesh for texture mapping (box projection)."""
        if trimesh is None:
            return mesh
        if not hasattr(mesh, "vertices") or mesh.vertices is None:
            return mesh
        # Simple box projection: store UV in vertex attributes if supported
        # Many exporters (e.g. OBJ) will write UV when present on visual
        try:
            verts = mesh.vertices
            uvs = []
            for v in verts:
                x, y, z = v[0], v[1], v[2]
                u = (x + 10) / 20 % 1.0
                v_ = (z + 10) / 20 % 1.0
                uvs.append([u, v_])
            if hasattr(trimesh.visual, "TextureVisuals"):
                mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs))
        except Exception:
            pass
        return mesh
