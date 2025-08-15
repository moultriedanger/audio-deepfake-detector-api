import os
import torch
import librosa
from transformers import (
    AutoModelForAudioClassification,
    AutoProcessor,
    AutoFeatureExtractor,
)

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mo-thecreator/Deepfake-audio-detection")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/deepfake_model")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache") 
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TARGET_SR = 16000

_processor = None  
_model = None

def _load_from(path_or_id: str):
    """
    Load processor/extractor + model from a local dir (if present)
    or from the HF Hub model id. Works with either Processor or FeatureExtractor.
    """
    # Try AutoProcessor first; fall back to AutoFeatureExtractor if needed
    try:
        proc = AutoProcessor.from_pretrained(path_or_id)
    except Exception:
        proc = AutoFeatureExtractor.from_pretrained(path_or_id)

    model = AutoModelForAudioClassification.from_pretrained(path_or_id)
    return proc, model

def _ensure_loaded():
    global _processor, _model
    if _processor is not None and _model is not None:
        return
    # Prefer local dir if it has files; otherwise use HF Hub
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        _processor, _model = _load_from(MODEL_DIR)
    else:
        _processor, _model = _load_from(HF_MODEL_ID)

def run_inference(file_path: str):
    _ensure_loaded()

    # librosa (+audioread + ffmpeg) supports wav/mp3/m4a/flac/ogg
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Works for both Processor and FeatureExtractor
    inputs = _processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")

    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_id = int(probs.argmax().item())

    id2label = getattr(_model.config, "id2label", {}) or {}
    label = id2label.get(pred_id, str(pred_id))

    return {
        "prediction": label,
        "scores": {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs.tolist())},
    }
