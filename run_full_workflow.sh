#!/bin/bash
set -e  # Dừng ngay nếu có lệnh bị lỗi

# Tạo thư mục chứa model
mkdir -p model

echo "========================================================"
echo ">>> STEP 1: TRAIN MODEL 1"
echo "========================================================"
python3 weather_train_1.py \
    --data-path s3://${DATA_BUCKET}/data/weather_dataset.csv \
    --model-output-path model/model_1.pth

echo "========================================================"
echo ">>> STEP 2: TRAIN MODEL 2"
echo "========================================================"
python3 weather_train_2.py \
    --data-path s3://${DATA_BUCKET}/data/weather_dataset.csv \
    --model-output-path model/model_2.pth

echo "========================================================"
echo ">>> STEP 3: EVALUATE (TEST 1, 2, 3)"
echo "========================================================"
# Test 1
python3 weather_test_1.py \
    --data-path s3://${DATA_BUCKET}/data/weather_test_1.csv \
    --model1-path model/model_1.pth \
    --model2-path model/model_2.pth \
    --out-dir model

# Test 2
python3 weather_test_2.py \
    --data-path s3://${DATA_BUCKET}/data/weather_test_2.csv \
    --model1-path model/model_1.pth \
    --model2-path model/model_2.pth \
    --out-dir model

# Test 3
python3 weather_test_3.py \
    --data-path s3://${DATA_BUCKET}/data/weather_test_3.csv \
    --model1-path model/model_1.pth \
    --model2-path model/model_2.pth \
    --out-dir model

echo "========================================================"
echo ">>> STEP 4: SELECT BEST MODEL"
echo "========================================================"
python3 select_best_model.py \
    --results-dir model \
    --out-dir model

echo ">>> WORKFLOW COMPLETED SUCCESSFULLY"