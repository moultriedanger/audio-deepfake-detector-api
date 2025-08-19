# app/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app(testing: bool = True):
    app = Flask(__name__)
    
    ALLOWED_ORIGINS = ["https://audio-deepfake-detector.vercel.app"]
    
    CORS(
        app,
        resources={r"/*": {"origins": ALLOWED_ORIGINS}},
        supports_credentials=True,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"]
    )

    USE_S3 = os.getenv("USE_S3_DOWNLOAD", "false").lower() == "true"

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    S3_BUCKET = os.getenv("S3_BUCKET", "deepfake-detector-model-storage")
    S3_PREFIX = os.getenv("S3_PREFIX", "deepfake_model/")
    MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "/tmp/deepfake_model"))

    # Ensure local model folder exists *before* we import inference
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        if USE_S3:
            from .model_loader import download_dir_from_s3
            os.makedirs(MODEL_DIR, exist_ok=True)
            download_dir_from_s3(S3_BUCKET, S3_PREFIX, MODEL_DIR)
        # else: inference will load from HF hub at runtime

    # Import after the folder is present (or skipped)
    from .model.inference import run_inference

    @app.route("/")
    def home():
        return "ok"

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file"}), 400

        f = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        try:
            result = run_inference(path)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
