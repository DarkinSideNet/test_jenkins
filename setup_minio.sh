#!/bin/bash
# Dừng script nếu có bất kỳ lỗi nào xảy ra
set -e

echo "--- [1] CONFIGURING MINIO ---"

mcli alias set myminio https://minio.neikoscloud.net admin admin123

echo "--- [2] SEARCHING FOR LATEST FILES ---"
# Sử dụng awk để lấy chính xác tên file từ JSON
DATA_FILE=$(mcli ls myminio/devopsproject/dataset_test/ --json | tail -n 1 | awk -F'"key":"' '{print $2}' | awk -F'"' '{print $1}')
MODEL_FILE=$(mcli ls myminio/devopsproject/current_model/ --json | tail -n 1 | awk -F'"key":"' '{print $2}' | awk -F'"' '{print $1}')

echo "Found Data: [$DATA_FILE]"
echo "Found Model: [$MODEL_FILE]"

if [ -z "$DATA_FILE" ] || [ -z "$MODEL_FILE" ]; then
    echo "❌ Error: Cannot find files on MinIO. Checking bucket list..."
    mcli ls myminio/devopsproject/
    exit 1
fi

echo "--- [3] DOWNLOADING ---"
mcli cp "myminio/devopsproject/dataset_test/$DATA_FILE" ./dataset.csv
mcli cp "myminio/devopsproject/current_model/$MODEL_FILE" ./model.pth

echo "--- [4] VERIFY ---"
ls -lh dataset.csv model.pth
echo "✅ Everything is ready!"