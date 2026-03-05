"""Constants used across ReconStruct pipeline."""

# Element type labels for detection
WALL_LABEL = "wall"
DOOR_LABEL = "door"
WINDOW_LABEL = "window"
ROOM_LABEL = "room"
STAIRS_LABEL = "stairs"
FIXTURE_LABEL = "fixture"

# Graph / mesh keys
NODE_TYPE_WALL = "wall"
NODE_TYPE_ROOM = "room"
NODE_TYPE_DOOR = "door"
NODE_TYPE_WINDOW = "window"
EDGE_TYPE_WALL = "wall"
EDGE_TYPE_OPENING = "opening"

# Image processing
GRAYSCALE = "grayscale"
BINARY = "binary"
RGB = "rgb"

# Export formats
EXPORT_OBJ = "obj"
EXPORT_FBX = "fbx"
EXPORT_GLTF = "gltf"
EXPORT_GLB = "glb"

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
