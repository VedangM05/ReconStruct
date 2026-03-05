"""Integration tests for ReconStruct spec pipeline."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processing.preprocessor import BlueprintPreprocessor
from src.image_processing.wall_detector import WallDetector
from src.mesh_generation.graph_builder import FloorPlanGraph
from src.mesh_generation.mesh_extrusion import MeshExtruder


@pytest.fixture
def sample_image():
    """Create sample binary image for testing."""
    image = np.zeros((500, 500), dtype=np.uint8)
    image[100:400, 100] = 255  # Vertical wall
    image[400, 100:400] = 255  # Horizontal wall
    return image


def test_wall_detection(sample_image):
    detector = WallDetector()
    walls = detector.detect_walls(sample_image)
    assert len(walls) > 0, "Should detect walls"


def test_graph_building(sample_image):
    detector = WallDetector()
    walls = detector.detect_walls(sample_image)

    graph_builder = FloorPlanGraph()
    graph = graph_builder.build_graph(walls)

    assert graph.number_of_nodes() > 0, "Graph should have nodes"
    assert graph.number_of_edges() > 0, "Graph should have edges"


def test_mesh_extrusion(sample_image):
    detector = WallDetector()
    walls = detector.detect_walls(sample_image)

    graph_builder = FloorPlanGraph()
    graph_builder.build_graph(walls)
    graph_builder.detect_rooms()
    graph_data = graph_builder.get_graph_data()

    extruder = MeshExtruder()
    mesh = extruder.create_complete_mesh(graph_data)

    assert mesh is not None, "Should create mesh"
    assert len(mesh.vertices) > 0, "Mesh should have vertices"
