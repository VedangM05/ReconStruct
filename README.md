# ReconStruct: 2D Blueprint to 3D Model Conversion System

Fully automated pipeline that converts 2D architectural blueprints into interactive 3D building models. The system processes floor plans through image analysis, structural detection, mesh generation, and 3D rendering.

**Primary Goal:** Transform a blueprint image → extract architectural elements → generate 2D floor graph → extrude to 3D → render interactive environment.

## Project Structure

```
ReconStruct/
├── config/           # Paths, settings, constants
├── src/
│   ├── image_processing/   # Preprocessing, wall/object detection, features
│   ├── mesh_generation/    # Graph, floor plan parsing, extrusion, CSG
│   ├── visualization/      # Blender export, assets, textures, lighting
│   ├── rendering/          # Unity, web, VR export
│   └── utils/              # Validators, logger, performance
├── models/           # Trained models (wall_detector, object_detector)
├── assets/           # 3D objects, textures, materials
├── tests/
└── data/             # Input/intermediate data
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

## Environment

Create a `.env` file (optional):

- `LOG_LEVEL` – Logging level (default: INFO)
- `RENDER_ENGINE` – unity | unreal | web
- `ENABLE_VR` – true | false

## Usage

```bash
reconstruct path/to/blueprint.png
```

Or from Python:

```python
from src.main import run_pipeline

run_pipeline("path/to/blueprint.png", output_dir="output/")
```

## Pipeline Stages

1. **Image processing** – Normalize resolution, denoise, segment
2. **Detection** – Walls, doors, windows, rooms
3. **Graph** – 2D floor plan graph from detected elements
4. **Mesh** – Extrude walls, apply openings, CSG
5. **Visualization** – Textures, lighting, asset placement
6. **Export** – Unity, web (GLTF/GLB), or VR

## License

MIT
