"""
ReconStruct: 2D Blueprint to 3D Model Conversion Pipeline.
Entry point: process blueprint image -> 3D mesh -> export.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import ensure_dirs, OUTPUT_DIR
from config.settings import RENDER_ENGINE, ENABLE_VR, PIXEL_TO_METER
from src.utils.validators import validate_blueprint_path
from src.utils.logger import get_logger
from src.utils.performance_monitor import PerformanceMonitor
from src.image_processing.preprocessor import BlueprintPreprocessor
from src.image_processing.feature_extractor import FeatureExtractor
from src.image_processing.wall_detector import WallDetector
from src.image_processing.object_detector import ObjectDetector
from src.mesh_generation.graph_builder import FloorPlanGraphBuilder, FloorPlanGraph
from src.mesh_generation.floor_plan_parser import FloorPlanParser
from src.mesh_generation.mesh_extrusion import MeshExtruder
from src.visualization.blender_exporter import BlenderExporter
from src.visualization.asset_placer import AssetPlacer
from src.rendering.web_exporter import WebExporter
from src.rendering.unity_exporter import UnityExporter
from src.rendering.vr_handler import VRHandler

logger = get_logger("reconstruct")


def run_pipeline(
    blueprint_path: str,
    output_dir: str = None,
    *,
    export_formats: list = None,
    wall_height: float = None,
    denoise: bool = True,
) -> dict:
    """
    Run full pipeline: load blueprint -> preprocess -> detect -> graph -> parse -> extrude -> export.
    Returns dict with keys: success, mesh, output_paths, features, perf_summary.
    """
    ensure_dirs()
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    perf = PerformanceMonitor()
    result = {"success": False, "mesh": None, "output_paths": [], "features": None, "perf_summary": {}}

    ok, err = validate_blueprint_path(blueprint_path)
    if not ok:
        logger.error(err)
        return result

    try:
        with perf.stage("preprocess"):
            preprocessor = BlueprintPreprocessor()
            preprocessed = preprocessor.run(blueprint_path, denoise=denoise, binarize_output=True)
        image = preprocessed["image"]
        binary = preprocessed["binary"]
        gray = preprocessed["gray"]

        with perf.stage("feature_extraction"):
            extractor = FeatureExtractor()
            features = extractor.extract(image, binary, detect_rooms=True)
        result["features"] = features

        if not features.walls:
            logger.warning("No walls detected; mesh may be empty.")

        with perf.stage("graph_build"):
            graph_builder = FloorPlanGraphBuilder()
            graph = graph_builder.build(
                features.walls,
                doors=features.doors,
                windows=features.windows,
                image_shape=features.image_shape,
            )

        with perf.stage("parse"):
            parser = FloorPlanParser(pixel_to_meter=PIXEL_TO_METER)
            plan = parser.parse(features, graph=graph)

        with perf.stage("extrude"):
            extruder = MeshExtruder(wall_height=wall_height)
            mesh = extruder.extrude_floor_plan(plan, combine=True)
        result["mesh"] = mesh

        if mesh is not None:
            export_formats = export_formats or ["obj", "glb"]
            with perf.stage("export"):
                exporter = BlenderExporter(output_dir=str(out_dir))
                base = Path(blueprint_path).stem
                paths = exporter.export_scene([mesh], base_name=base, formats=export_formats)
                result["output_paths"] = [str(p) for p in paths]
            if RENDER_ENGINE == "web":
                web = WebExporter(output_dir=str(out_dir))
                web.export_glb(mesh, str(out_dir / f"{base}.glb"))
            elif RENDER_ENGINE == "unity":
                u = UnityExporter(output_dir=str(out_dir))
                u.export_mesh_fbx(mesh, str(out_dir / f"{base}.obj"))
                scene = u.build_scene(
                    mesh_paths=result["output_paths"],
                    lights=[],
                )
                u.export_scene_json(scene, name=base)
            if ENABLE_VR:
                vr = VRHandler(enabled=True)
                logger.info("VR manifest: %s", vr.vr_manifest())
        else:
            logger.warning("No mesh produced; skipping export.")

        result["perf_summary"] = perf.summary()
        result["success"] = True
        logger.info("Pipeline finished. Outputs: %s", result["output_paths"])
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        result["success"] = False
    return result


class ReconStructPipeline:
    """
    Complete ReconStruct pipeline orchestrator (spec API).
    Manages all stages from image input to 3D model output.
    """

    def __init__(self):
        self.preprocessor = BlueprintPreprocessor()
        self.wall_detector = WallDetector()
        self.object_detector = ObjectDetector()
        self.performance_monitor = PerformanceMonitor()

    def process_blueprint(self, image_path: str, output_dir: str = None):
        """Execute complete pipeline. Returns dict with floor_plan, mesh_path, objects_detected."""
        import cv2
        out_dir = output_dir or str(OUTPUT_DIR)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("ReconStruct Pipeline Starting")
        logger.info("=" * 60)

        with self.performance_monitor.track("full_pipeline"):
            logger.info("\n--- PHASE 1: IMAGE PREPROCESSING ---")
            with self.performance_monitor.track("preprocessing"):
                original, gray, binary, processed = self.preprocessor.preprocess_full_pipeline(image_path)
                self.preprocessor.save_intermediate_results(out_dir, "debug", original, gray, binary)

            logger.info("\n--- PHASE 2: WALL DETECTION ---")
            with self.performance_monitor.track("wall_detection"):
                walls = self.wall_detector.detect_walls(binary, debug_output_dir=out_dir)
                viz_walls = self.wall_detector.visualize_walls(processed, walls)
                cv2.imwrite(f"{out_dir}/03_walls_detected.png", viz_walls)

            logger.info("\n--- PHASE 3: OBJECT DETECTION ---")
            with self.performance_monitor.track("object_detection"):
                objects = self.object_detector.detect_objects(processed)
                filtered_objects = self.object_detector.filter_detections(objects)
                viz_objects = self.object_detector.visualize_detections(processed, filtered_objects)
                cv2.imwrite(f"{out_dir}/04_objects_detected.png", viz_objects)

            logger.info("\n--- PHASE 4: 2D GRAPH GENERATION ---")
            with self.performance_monitor.track("graph_construction"):
                graph_builder = FloorPlanGraph()
                graph_builder.build_graph(walls)
                graph_builder.detect_rooms()
                graph_data = graph_builder.get_graph_data()
                graph_file = Path(out_dir) / "floor_plan.json"
                with open(graph_file, "w") as f:
                    json.dump(graph_data, f, indent=2)

            logger.info("\n--- PHASE 5: 3D MESH EXTRUSION ---")
            with self.performance_monitor.track("mesh_extrusion"):
                extruder = MeshExtruder()
                mesh_3d = extruder.create_complete_mesh(graph_data)
                mesh_file = Path(out_dir) / "building_mesh.obj"
                if mesh_3d is not None:
                    extruder.export_mesh(str(mesh_file))

            logger.info("\n--- PHASE 6: ASSET PLACEMENT ---")
            doors = [o for o in filtered_objects if o.get("class") == "door"]
            windows = [o for o in filtered_objects if o.get("class") == "window"]
            # Convert to meters and center (same as mesh) so assets align with building
            nodes = graph_data.get("nodes", [])
            if nodes:
                cx = sum(n["x"] for n in nodes) / len(nodes)
                cy = sum(n["y"] for n in nodes) / len(nodes)
            else:
                cx = cy = 0
            for d in doors:
                x1, y1, x2, y2 = d["bbox"]
                px, py = (x1 + x2) / 2, (y1 + y2) / 2
                d["location"] = ((px - cx) * PIXEL_TO_METER, (py - cy) * PIXEL_TO_METER)
            for w in windows:
                x1, y1, x2, y2 = w["bbox"]
                px, py = (x1 + x2) / 2, (y1 + y2) / 2
                w["location"] = ((px - cx) * PIXEL_TO_METER, (py - cy) * PIXEL_TO_METER)

            final_mesh = mesh_3d
            if final_mesh is not None:
                with self.performance_monitor.track("asset_placement"):
                    placer = AssetPlacer()
                    if doors:
                        final_mesh = placer.place_doors(final_mesh, doors)
                    if windows:
                        final_mesh = placer.place_windows(final_mesh, windows)

            logger.info("\n--- PHASE 7: EXPORT & RENDERING ---")
            with self.performance_monitor.track("export"):
                if final_mesh is not None:
                    final_mesh.export(str(Path(out_dir) / "final_model.glb"))
                    final_mesh.export(str(Path(out_dir) / "final_model.obj"))
                UnityExporter.export_scene(graph_data, out_dir)

        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        self.performance_monitor.print_summary()
        logger.info("\nPipeline Complete!")
        logger.info("Output saved to: %s", out_dir)

        return {
            "floor_plan": graph_data,
            "mesh_path": str(Path(out_dir) / "final_model.glb"),
            "objects_detected": {"doors": len(doors), "windows": len(windows)},
        }


def main():
    parser = argparse.ArgumentParser(description="ReconStruct: 2D Blueprint to 3D Model")
    parser.add_argument("blueprint", help="Path to blueprint image")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--legacy", action="store_true", help="Use legacy pipeline (run_pipeline)")
    parser.add_argument("--format", action="append", dest="formats", default=[], help="Export format (legacy)")
    parser.add_argument("--wall-height", type=float, default=None, help="Wall height in meters (legacy)")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising (legacy)")
    args = parser.parse_args()

    if args.legacy:
        formats = args.formats if args.formats else ["obj", "glb"]
        result = run_pipeline(
            args.blueprint,
            output_dir=args.output,
            export_formats=formats,
            wall_height=args.wall_height,
            denoise=not args.no_denoise,
        )
        sys.exit(0 if result["success"] else 1)

    if not Path(args.blueprint).exists():
        print(f"Error: Blueprint file not found: {args.blueprint}")
        print("Please provide a path to an existing image (e.g. PNG or JPG).")
        print("Example: python3 src/main.py ./data/input/my_floor_plan.png -o ./results")
        sys.exit(1)

    pipeline = ReconStructPipeline()
    result = pipeline.process_blueprint(args.blueprint, args.output)
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
