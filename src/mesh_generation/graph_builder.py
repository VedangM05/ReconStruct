"""Build 2D floor plan graph from walls and openings."""

from typing import List, Tuple, Dict, Any
import math

import numpy as np
import networkx as nx

try:
    from config import settings
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import settings

from config.constants import NODE_TYPE_WALL, NODE_TYPE_ROOM, NODE_TYPE_DOOR, NODE_TYPE_WINDOW, EDGE_TYPE_WALL, EDGE_TYPE_OPENING
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class FloorPlanGraph:
    """
    Represents the 2D floor plan as a graph with nodes (intersections) and edges (walls).
    Forms the foundation for 3D mesh generation (spec pipeline).
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: List[Tuple[float, float]] = []
        self.edges: List[Dict[str, Any]] = []
        self.rooms: List[Dict[str, Any]] = []

    def extract_nodes_from_walls(
        self,
        wall_segments: np.ndarray,
        merge_distance: float = 20,
    ) -> np.ndarray:
        """
        Extract nodes from wall endpoints and intersections.
        Merges endpoints that are close together (within merge_distance).
        """
        if len(wall_segments) == 0:
            logger.warning("No wall segments provided")
            return np.array([])
        endpoints = []
        for seg in wall_segments:
            x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
            endpoints.append([x1, y1])
            endpoints.append([x2, y2])
        endpoints = np.array(endpoints)
        unique_nodes = []
        used = set()
        for i, point in enumerate(endpoints):
            if i in used:
                continue
            distances = np.linalg.norm(endpoints - point, axis=1)
            close_points = np.where(distances < merge_distance)[0]
            merged_point = endpoints[close_points].mean(axis=0)
            unique_nodes.append(merged_point)
            for idx in close_points:
                used.add(idx)
        nodes = np.array(unique_nodes)
        logger.info("Extracted %d unique nodes from walls", len(nodes))
        return nodes

    def build_edges_from_walls(
        self,
        wall_segments: np.ndarray,
        nodes: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Connect nodes based on wall segments.
        Creates edges representing walls in the graph.
        """
        edges = []
        tolerance = 30
        for seg in wall_segments:
            x1, y1, x2, y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
            start_point = np.array([x1, y1])
            end_point = np.array([x2, y2])
            distances_start = np.linalg.norm(nodes - start_point, axis=1)
            distances_end = np.linalg.norm(nodes - end_point, axis=1)
            start_node_id = int(np.argmin(distances_start))
            end_node_id = int(np.argmin(distances_end))
            if (
                distances_start[start_node_id] < tolerance
                and distances_end[end_node_id] < tolerance
            ):
                wall_length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                edges.append({
                    "start_node": start_node_id,
                    "end_node": end_node_id,
                    "length": wall_length,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        logger.info("Created %d edges from walls", len(edges))
        return edges

    def build_graph(self, wall_segments: np.ndarray) -> nx.Graph:
        """Build complete floor plan graph from wall segments."""
        nodes = self.extract_nodes_from_walls(wall_segments)
        self.nodes = [tuple(p) for p in nodes]
        for i, (x, y) in enumerate(self.nodes):
            self.graph.add_node(i, pos=(float(x), float(y)))
        edges = self.build_edges_from_walls(wall_segments, nodes)
        self.edges = edges
        for edge in edges:
            start, end = edge["start_node"], edge["end_node"]
            if start != end:
                self.graph.add_edge(
                    start, end,
                    length=edge["length"],
                    wall_data=edge,
                )
        logger.info(
            "Built graph with %d nodes and %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def detect_rooms(self) -> List[Dict[str, Any]]:
        """
        Detect rooms using cycle detection in the graph.
        Each cycle represents a closed room boundary.
        """
        try:
            cycles = nx.cycle_basis(self.graph)
        except Exception:
            cycles = []
        rooms = []
        for cycle in cycles:
            if len(cycle) >= 3:
                room_nodes = [self.nodes[node_id] for node_id in cycle]
                polygon = np.array(room_nodes)
                area = self._polygon_area(polygon)
                if area > 100:
                    rooms.append({
                        "nodes": cycle,
                        "polygon": polygon,
                        "area": area,
                    })
        self.rooms = rooms
        logger.info("Detected %d rooms", len(rooms))
        return rooms

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula."""
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def get_graph_data(self) -> Dict[str, Any]:
        """Export graph data as dictionary for serialization."""
        return {
            "nodes": [
                {"id": i, "x": float(x), "y": float(y)}
                for i, (x, y) in enumerate(self.nodes)
            ],
            "edges": self.edges,
            "rooms": [
                {
                    "nodes": room["nodes"],
                    "polygon": room["polygon"].tolist() if hasattr(room["polygon"], "tolist") else room["polygon"],
                    "area": float(room["area"]),
                }
                for room in self.rooms
            ],
        }


def _segment_endpoints(x1: float, y1: float, x2: float, y2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ((x1, y1), (x2, y2))


class FloorPlanGraphBuilder:
    """Build a NetworkX graph representing the floor plan: nodes = corners/vertices, edges = walls/openings."""

    def __init__(self, pixel_to_meter: float = None):
        self.pixel_to_meter = pixel_to_meter or settings.PIXEL_TO_METER

    def _to_meters(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.pixel_to_meter, y * self.pixel_to_meter)

    def _snap_to_existing(
        self,
        point: Tuple[float, float],
        existing: List[Tuple[float, float]],
        tol: float = 5.0,
    ) -> Tuple[float, float]:
        """Snap point to existing vertex if within tolerance (pixels)."""
        px, py = point
        for (ex, ey) in existing:
            if _dist((px, py), (ex, ey)) <= tol:
                return (ex, ey)
        return (px, py)

    def build(
        self,
        walls: List[Tuple[float, float, float, float]],
        doors: List[Any] = None,
        windows: List[Any] = None,
        image_shape: Tuple[int, int] = (0, 0),
        snap_tolerance_px: float = 5.0,
    ) -> nx.Graph:
        """
        Build graph from wall segments and optional door/window bboxes.
        Nodes have attributes: type (wall_corner, room, door, window), x, y (meters).
        Edges have attributes: type (wall, opening), length_m.
        """
        G = nx.Graph()
        doors = doors or []
        windows = windows or []
        vertices = []
        # Collect all segment endpoints and snap
        for (x1, y1, x2, y2) in walls:
            p1 = self._snap_to_existing((x1, y1), vertices, snap_tolerance_px)
            p2 = self._snap_to_existing((x2, y2), vertices, snap_tolerance_px)
            if p1 not in [v for v in vertices]:
                vertices.append(p1)
            if p2 not in [v for v in vertices]:
                vertices.append(p2)
            m1 = self._to_meters(p1[0], p1[1])
            m2 = self._to_meters(p2[0], p2[1])
            length_m = _dist(m1, m2)
            G.add_node(
                p1,
                type=NODE_TYPE_WALL,
                x_m=m1[0],
                y_m=m1[1],
                x_px=p1[0],
                y_px=p1[1],
            )
            G.add_node(
                p2,
                type=NODE_TYPE_WALL,
                x_m=m2[0],
                y_m=m2[1],
                x_px=p2[0],
                y_px=p2[1],
            )
            G.add_edge(
                p1,
                p2,
                type=EDGE_TYPE_WALL,
                length_m=length_m,
            )
        # Add door/window nodes at bbox centers
        for obj in doors:
            (x1, y1, x2, y2) = obj.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cm = self._to_meters(cx, cy)
            node_id = ("door", cx, cy)
            G.add_node(
                node_id,
                type=NODE_TYPE_DOOR,
                x_m=cm[0],
                y_m=cm[1],
                x_px=cx,
                y_px=cy,
                width_px=max(x2 - x1, 1),
                height_px=max(y2 - y1, 1),
            )
        for obj in windows:
            (x1, y1, x2, y2) = obj.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cm = self._to_meters(cx, cy)
            node_id = ("window", cx, cy)
            G.add_node(
                node_id,
                type=NODE_TYPE_WINDOW,
                x_m=cm[0],
                y_m=cm[1],
                x_px=cx,
                y_px=cy,
                width_px=max(x2 - x1, 1),
                height_px=max(y2 - y1, 1),
            )
        return G
