from flask import Flask, request, jsonify
from .model.inference import run_inference
import os
from flask_cors import CORS
from dotenv import load_dotenv
import boto3
from .model_loader import download_model_if_missing


def create_app(testing: bool = True):

    app = Flask(__name__)
    CORS(app)

    UPLOAD_FOLDER = "uploads"
    MODEL_PATH = "model/librifake_pretrained_lambda0.5_epoch_25.pth"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    download_model_if_missing('deepfake-detector-model-storage', 
                            'librifake_pretrained_lambda0.5_epoch_25.pth', 
                            "model/librifake_pretrained_lambda0.5_epoch_25.pth")
    
    @app.route("/")
    def test_home():
        return f"hello world! {testing}"
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        try:
            result = run_inference(input_path, MODEL_PATH)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    return app


# if __name__ == "__main__":
#     app.run(
#         debug=False,
#         host="0.0.0.0",
#         port=int(os.environ.get("PORT", 5000))
#     )