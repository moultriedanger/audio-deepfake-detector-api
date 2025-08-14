# load.py
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import subprocess
from dotenv import load_dotenv
import os

load_dotenv()

model_id = "mo-thecreator/Deepfake-audio-detection"
out_dir = "deepfake_model"

m = AutoModelForAudioClassification.from_pretrained(model_id)
fe = AutoFeatureExtractor.from_pretrained(model_id)

m.save_pretrained(out_dir)
fe.save_pretrained(out_dir)

bucket = os.getenv("S3_BUCKET", "deepfake-detector-model-storage")
prefix = os.getenv("S3_PREFIX", "deepfake_model")
subprocess.run(
    ["aws", "s3", "cp", out_dir, f"s3://{bucket}/{prefix}", "--recursive"],
    check=True
)

print("Uploaded model to s3://%s/%s/" % (bucket, prefix))
