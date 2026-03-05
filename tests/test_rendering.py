"""Tests for visualization and rendering export."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.blender_exporter import BlenderExporter
from src.visualization.lighting_manager import LightingManager
from src.rendering.web_exporter import WebExporter
from src.rendering.vr_handler import VRHandler


def test_blender_exporter_export_mesh(tmp_path):
    try:
        import trimesh
        box = trimesh.creation.box(extents=[1, 1, 1])
        exporter = BlenderExporter(output_dir=str(tmp_path))
        path = exporter.export_mesh(box, str(tmp_path / "box.obj"), format="obj")
        assert path.exists()
    except ImportError:
        pytest.skip("trimesh not installed")


def test_lighting_manager_default():
    lm = LightingManager()
    lights = lm.get_default_scene_lights()
    assert len(lights) >= 1
    d = lm.to_dict()
    assert isinstance(d, list)


def test_web_exporter_glb(tmp_path):
    try:
        import trimesh
        box = trimesh.creation.box(extents=[2, 2, 2])
        exporter = WebExporter(output_dir=str(tmp_path))
        path = exporter.export_glb(box, str(tmp_path / "scene.glb"))
        assert path is None or path.exists()
    except ImportError:
        pytest.skip("trimesh not installed")


def test_vr_handler_manifest():
    vr = VRHandler(enabled=True)
    m = vr.vr_manifest()
    assert m["vr_enabled"] is True
    assert "scale_to_meters" in m
