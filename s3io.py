import os
import boto3
import pandas as pd
from io import StringIO

def get_s3_client():
    """
    Tạo S3 client. Nếu có biến môi trường S3_ENDPOINT_URL, 
    nó sẽ kết nối đến đó (MinIO), ngược lại kết nối AWS S3 chuẩn.
    """
    endpoint = os.environ.get("S3_ENDPOINT_URL", None)
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # Tạo client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint
    )
    return s3

def read_csv(s3_path):
    """
    Đọc file CSV từ S3/MinIO path: s3://bucket/key
    """
    client = get_s3_client()
    
    if not s3_path.startswith("s3://"):
        raise ValueError("Path must start with s3://")
    
    # Tách bucket và key
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket_name = path_parts[0]
    key = path_parts[1]
    
    print(f"[s3io] Downloading s3://{bucket_name}/{key} ...")
    if os.environ.get("S3_ENDPOINT_URL"):
        print(f"[s3io] Using Endpoint: {os.environ.get('S3_ENDPOINT_URL')}")

    obj = client.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(obj['Body'])