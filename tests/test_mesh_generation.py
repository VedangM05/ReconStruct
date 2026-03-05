"""Tests for mesh generation (graph, parse, extrusion)."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processing.feature_extractor import FloorPlanFeatures
from src.mesh_generation.graph_builder import FloorPlanGraphBuilder
from src.mesh_generation.floor_plan_parser import FloorPlanParser
from src.mesh_generation.mesh_extrusion import MeshExtruder


@pytest.fixture
def sample_features():
    walls = [(0, 0, 10, 0), (10, 0, 10, 10), (10, 10, 0, 10), (0, 10, 0, 0)]
    return FloorPlanFeatures(
        walls=walls,
        doors=[],
        windows=[],
        fixtures=[],
        rooms=[],
        image_shape=(100, 100),
    )


def test_graph_builder_build(sample_features):
    builder = FloorPlanGraphBuilder(pixel_to_meter=0.01)
    G = builder.build(sample_features.walls, image_shape=(100, 100))
    assert G is not None
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1


def test_floor_plan_parser_parse(sample_features):
    parser = FloorPlanParser(pixel_to_meter=0.01)
    plan = parser.parse(sample_features)
    assert plan is not None
    assert len(plan.walls) == 4
    for w in plan.walls:
        assert w.length_m >= 0


def test_mesh_extruder_extrude(sample_features):
    parser = FloorPlanParser(pixel_to_meter=0.01)
    plan = parser.parse(sample_features)
    extruder = MeshExtruder(wall_height=3.0, wall_thickness=0.2)
    mesh = extruder.extrude_floor_plan(plan, combine=True)
    if mesh is not None:
        assert hasattr(mesh, "vertices")
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
