import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from .model_loader import download_dir_from_s3

load_dotenv()

def create_app(testing: bool = True):
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    S3_BUCKET = os.getenv("S3_BUCKET", "deepfake-detector-model-storage")
    S3_PREFIX = os.getenv("S3_PREFIX", "deepfake_model/")
    MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "model/deepfake_model"))

    # Ensure local model folder exists *before* we import inference
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_dir_from_s3(S3_BUCKET, S3_PREFIX, MODEL_DIR)

    # Import after the folder is present
    from .model.inference import run_inference

    @app.route("/")
    def home():
        return f"hello world! {testing}"

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file"}), 400

        f = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        try:
            result = run_inference(path)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
