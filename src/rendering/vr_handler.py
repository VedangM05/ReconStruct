"""VR-specific export and configuration (WebXR, Unity XR)."""

from pathlib import Path
from typing import Dict, Any, Optional


class VRHandler:
    """Configure and export VR-ready scene (scale, teleport areas, interaction)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def vr_manifest(self) -> Dict[str, Any]:
        """Return manifest for VR runtime (scale, units, required extensions)."""
        return {
            "vr_enabled": self.enabled,
            "scale_to_meters": 1.0,
            "teleport_floor": True,
            "required_extensions": ["WEBXR"],
        }

    def configure_for_webxr(self, scene_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add WebXR-related options to scene config."""
        if not self.enabled:
            return scene_config
        out = dict(scene_config)
        out["vr"] = self.vr_manifest()
        return out
