# app/model/inference.py
import os
from typing import Dict, Any, List
from transformers import pipeline

# Build absolute path from env (falls back to repo-root/model/deepfake_model)
DEFAULT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "deepfake_model")
)
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", DEFAULT_DIR))

_classifier = None  # lazy singleton

def _get_classifier():
    global _classifier
    if _classifier is None:
        # Sanity check so we fail clearly if the folder is missing/empty
        if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
            raise RuntimeError(
                f"MODEL_DIR '{MODEL_DIR}' missing or empty. "
                "Ensure S3 download ran and/or MODEL_DIR points to the correct local folder."
            )
        _classifier = pipeline(
            task="audio-classification",
            model=MODEL_DIR,
            device=-1,
            top_k=None
        )
    return _classifier

def run_inference(file_path: str) -> Dict[str, Any]:
    clf = _get_classifier()
    outputs: List[dict] = clf(file_path)
    scores = {o["label"].lower(): float(o["score"]) for o in outputs}
    fake_score = scores.get("fake", 0.0)
    real_score = scores.get("real", 0.0)
    label = "fake" if fake_score >= real_score else "real"
    return {"label": label, "scores": {"fake": fake_score, "real": real_score}}
