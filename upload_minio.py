import os
import boto3
from datetime import datetime

# --- CẤU HÌNH MINIO ---
MINIO_URL = "https://minio.neikoscloud.net" # Thay bằng IP/Domain của bạn
ACCESS_KEY = "admin"
SECRET_KEY = "admin123"
BUCKET_NAME = "devopsproject"
BASE_PATH = "data_all_train"

# Khởi tạo S3 Client
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

def upload_folders_to_minio():
    # 1. Tạo tên thư mục dựa trên thời gian hiện tại (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Danh sách các thư mục cần upload
    folders_to_upload = ['top3_models_incremental', 'models_incremental', 'best_model_final', 'evaluation_logs', 'dataset_test', 'test_logs']
    prod_model_local = "best_model_final/weather_model_production.pth"
    static_s3_path = "current_model/model.pth"
    
    print(f"--- Start uploading the link: {BASE_PATH}/{timestamp} ---")

    for folder in folders_to_upload:
        if not os.path.isdir(folder):
            print(f"Warning: Folder {folder} not found, skipping...")
            continue
            
        # Duyệt qua tất cả các file trong thư mục (đệ quy)
        for root, dirs, files in os.walk(folder):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Tạo đường dẫn đích trên MinIO
                # Cấu trúc: data_all_train/20251218_174805/folder_name/file_name
                s3_path = f"{BASE_PATH}/{timestamp}/{local_path}".replace("\\", "/")
                
                try:
                    s3_client.upload_file(local_path, BUCKET_NAME, s3_path)
                    print(f"Uploaded: {local_path} -> {s3_path}")
                except Exception as e:
                    print(f"Lỗi khi upload {local_path}: {e}")

        if os.path.exists(prod_model_local):
            print(f"\n--- Updating the production model now: {static_s3_path} ---")
            try:
                s3_client.upload_file(prod_model_local, BUCKET_NAME, static_s3_path)
                print(f"Success: Overwritten {prod_model_local} -> {static_s3_path}")
            except Exception as e:
                print(f"Error overwriting current model: {e}")
        else:
            print(f"Warning: {prod_model_local} not found to update current version.")
    print("--- Upload pipeline completed ---")

if __name__ == "__main__":
    upload_folders_to_minio()