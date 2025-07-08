import os
from dotenv import load_dotenv
import boto3

load_dotenv()

def download_model_if_missing(bucket, key, local_path):
    if not os.path.exists(local_path):
        print("Downloading model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, local_path)
        print("Download complete.")
    else:
        print("Model already exists locally.")