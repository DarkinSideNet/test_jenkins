#!/bin/bash
set -e

# Khai báo các đường dẫn MinIO
MINIO_EVAL_DATA="myminio/devopsproject/evaluation_dataset/"
LOCAL_DATA_DIR="./evaluation_data"

echo "-------------------------------------------------------"
echo "STEP 1: CLEANING & DOWNLOADING DATA FROM MINIO"
echo "-------------------------------------------------------"
# Xóa dữ liệu cũ và tạo thư mục mới
rm -rf $LOCAL_DATA_DIR
mkdir -p $LOCAL_DATA_DIR

# Sử dụng mcli để kéo dữ liệu
mcli cp --recursive $MINIO_EVAL_DATA $LOCAL_DATA_DIR

echo "-------------------------------------------------------"
echo "STEP 2: RUNNING EVALUATION PYTHON SCRIPT"
echo "-------------------------------------------------------"
# Chạy file python để tính toán metrics
python3 evaluate_models.py

echo "-------------------------------------------------------"
echo "STEP 3: SELECT BEST MODEL & DEPLOY TO PRODUCTION"
echo "-------------------------------------------------------"
# Gọi script chọn model tốt nhất (Winner)
python3 select_best_model.py

echo ">>> PHASE 2 COMPLETED SUCCESSFULLY"