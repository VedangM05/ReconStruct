"""Parse floor plan features into structured representation for mesh generation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import networkx as nx

from src.image_processing.feature_extractor import FloorPlanFeatures


@dataclass
class WallSegment:
    """Single wall segment in metric space."""
    x1_m: float
    y1_m: float
    x2_m: float
    y2_m: float
    length_m: float
    openings: List[Dict[str, Any]] = field(default_factory=list)  # doors/windows on this segment


@dataclass
class ParsedFloorPlan:
    """Structured floor plan ready for extrusion."""
    walls: List[WallSegment]
    graph: Optional[nx.Graph] = None
    scale_px_to_m: float = 0.01


class FloorPlanParser:
    """Convert FloorPlanFeatures and optional graph into ParsedFloorPlan."""

    def __init__(self, pixel_to_meter: float = 0.01):
        self.pixel_to_meter = pixel_to_meter

    def _length(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _assign_openings_to_walls(
        self,
        walls: List[WallSegment],
        doors: List[Any],
        windows: List[Any],
        pixel_to_meter: float,
    ) -> None:
        """Assign each door/window to nearest wall segment (by perpendicular distance)."""
        for d in doors:
            (x1, y1, x2, y2) = d.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cm_x = cx * pixel_to_meter
            cm_y = cy * pixel_to_meter
            best_wall = None
            best_t = 0.5
            best_dist = 1e9
            for i, w in enumerate(walls):
                # Project (cm_x, cm_y) onto segment
                dx = w.x2_m - w.x1_m
                dy = w.y2_m - w.y1_m
                L = self._length(w.x1_m, w.y1_m, w.x2_m, w.y2_m)
                if L < 1e-6:
                    continue
                t = ((cm_x - w.x1_m) * dx + (cm_y - w.y1_m) * dy) / (L * L)
                t = max(0, min(1, t))
                proj_x = w.x1_m + t * dx
                proj_y = w.y1_m + t * dy
                dist = self._length(cm_x, cm_y, proj_x, proj_y)
                if dist < best_dist:
                    best_dist = dist
                    best_wall = i
                    best_t = t
            if best_wall is not None and best_dist < 0.5:
                walls[best_wall].openings.append({
                    "type": "door",
                    "t": best_t,
                    "width_m": (x2 - x1) * pixel_to_meter,
                    "height_m": getattr(d, "height_m", 2.1),
                })
        for d in windows:
            (x1, y1, x2, y2) = d.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cm_x = cx * pixel_to_meter
            cm_y = cy * pixel_to_meter
            best_wall = None
            best_t = 0.5
            best_dist = 1e9
            for i, w in enumerate(walls):
                dx = w.x2_m - w.x1_m
                dy = w.y2_m - w.y1_m
                L = self._length(w.x1_m, w.y1_m, w.x2_m, w.y2_m)
                if L < 1e-6:
                    continue
                t = ((cm_x - w.x1_m) * dx + (cm_y - w.y1_m) * dy) / (L * L)
                t = max(0, min(1, t))
                proj_x = w.x1_m + t * dx
                proj_y = w.y1_m + t * dy
                dist = self._length(cm_x, cm_y, proj_x, proj_y)
                if dist < best_dist:
                    best_dist = dist
                    best_wall = i
                    best_t = t
            if best_wall is not None and best_dist < 0.5:
                walls[best_wall].openings.append({
                    "type": "window",
                    "t": best_t,
                    "width_m": (x2 - x1) * pixel_to_meter,
                    "height_m": getattr(d, "height_m", 1.2),
                })

    def parse(
        self,
        features: FloorPlanFeatures,
        graph: Optional[nx.Graph] = None,
    ) -> ParsedFloorPlan:
        """Convert features (and optional graph) to ParsedFloorPlan with wall segments and openings."""
        walls = []
        for (x1, y1, x2, y2) in features.walls:
            x1_m = x1 * self.pixel_to_meter
            y1_m = y1 * self.pixel_to_meter
            x2_m = x2 * self.pixel_to_meter
            y2_m = y2 * self.pixel_to_meter
            length_m = self._length(x1_m, y1_m, x2_m, y2_m)
            walls.append(
                WallSegment(
                    x1_m=x1_m, y1_m=y1_m,
                    x2_m=x2_m, y2_m=y2_m,
                    length_m=length_m,
                    openings=[],
                )
            )
        self._assign_openings_to_walls(
            walls,
            features.doors,
            features.windows,
            self.pixel_to_meter,
        )
        return ParsedFloorPlan(
            walls=walls,
            graph=graph,
            scale_px_to_m=self.pixel_to_meter,
        )
