"""Flask API server for ReconStruct blueprint processing."""

import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path

from src.main import ReconStructPipeline
from config.paths import OUTPUT_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

pipeline = ReconStructPipeline()


@app.route("/api/process", methods=["POST"])
def process_blueprint():
    """Process uploaded blueprint."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = app.config["UPLOAD_FOLDER"] / filename
    file.save(str(filepath))

    try:
        result = pipeline.process_blueprint(str(filepath), output_dir=str(OUTPUT_DIR))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/<model_id>", methods=["GET"])
def get_model(model_id):
    """Download generated 3D model."""
    model_path = Path("output") / model_id / "final_model.glb"
    if not model_path.exists():
        model_path = OUTPUT_DIR / model_id / "final_model.glb"
    if model_path.exists():
        return send_file(str(model_path), mimetype="application/octet-stream")
    return jsonify({"error": "Model not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
