# app.py

from flask import Flask, request, jsonify
from model.inference import run_inference
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/librifake_pretrained_lambda0.5_epoch_25.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def test_home():
    return "hello world!"

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

if __name__ == "__main__":
    app.run(debug=True)