import os
from dotenv import load_dotenv
import boto3

load_dotenv()

def download_dir_from_s3(bucket: str, prefix: str, dest_dir: str) -> None:
    """
    Download all objects under `prefix/` into local `dest_dir`.
    Use this for Hugging Face models saved with save_pretrained().
    """
    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    os.makedirs(dest_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)