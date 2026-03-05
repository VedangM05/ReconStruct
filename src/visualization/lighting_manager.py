"""Manage lighting setup for exported scenes (directional, ambient, baked)."""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Light:
    """Single light definition for export."""
    type: str  # "directional", "point", "spot", "ambient"
    position: List[float] = field(default_factory=lambda: [0, 0, 5])
    direction: List[float] = field(default_factory=lambda: [0, 0, -1])
    intensity: float = 1.0
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


class LightingManager:
    """Define and export lighting for Unity/Web/Blender."""

    def __init__(self, preset: str = "default"):
        self.preset = preset
        self._lights = []

    def add_directional(self, direction: List[float] = None, intensity: float = 1.0) -> Light:
        d = direction or [0.5, -1.0, -0.5]
        norm = (d[0]**2 + d[1]**2 + d[2]**2) ** 0.5
        if norm > 1e-6:
            d = [d[0]/norm, d[1]/norm, d[2]/norm]
        light = Light(type="directional", direction=d, intensity=intensity)
        self._lights.append(light)
        return light

    def add_point(self, position: List[float], intensity: float = 1.0) -> Light:
        light = Light(type="point", position=position, intensity=intensity)
        self._lights.append(light)
        return light

    def add_ambient(self, intensity: float = 0.3) -> Light:
        light = Light(type="ambient", intensity=intensity)
        self._lights.append(light)
        return light

    def get_default_scene_lights(self) -> List[Light]:
        """Return a default set of lights for indoor floor plan."""
        self._lights = []
        self.add_ambient(0.3)
        self.add_directional([0.5, -1.0, -0.5], 0.8)
        self.add_point([0, 0, 2.5], 0.5)
        return list(self._lights)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize lights for JSON/Unity export."""
        return [
            {
                "type": l.type,
                "position": l.position,
                "direction": l.direction,
                "intensity": l.intensity,
                "color": l.color,
            }
            for l in (self._lights or self.get_default_scene_lights())
        ]
