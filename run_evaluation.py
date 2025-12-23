import os
import subprocess
import pandas as pd
import shutil
import mlflow

# Cấu hình
DATASET_DIR = "dataset_test"
LOG_DIR = "test_logs"
EVAL_LOG_DIR = "evaluation_logs"
SUMMARY_PATH = os.path.join(EVAL_LOG_DIR, "evaluation_summary.csv")

mlflow.set_tracking_uri("https://mlflow.neikoscloud.net")
mlflow.set_experiment("weather_evaluation")

# Bắt đầu Run chính cho Eval
with mlflow.start_run(run_name=f"Eval_Batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"):
    # 1. Làm sạch dữ liệu cũ
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    # 2. Quét dữ liệu và chạy test
    TEST_DATASETS = sorted([os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".csv")])
    
    for idx, data_path in enumerate(TEST_DATASETS, 1):
        case_name = f"case_{idx}"
        # subprocess vẫn chạy độc lập
        subprocess.run(["python3", "weather_test.py", "--data", data_path, "--out_name", case_name])

    # 3. Tổng hợp kết quả
    all_results = []
    for idx in range(1, len(TEST_DATASETS) + 1):
        csv_path = os.path.join(LOG_DIR, f"case_{idx}_result.csv")
        if os.path.exists(csv_path):
            all_results.append(pd.read_csv(csv_path))

    if all_results:
        df_detail = pd.concat(all_results, ignore_index=True)
        summary = df_detail.groupby("model")[["mae", "rmse"]].mean().reset_index().sort_values("rmse")
        summary.to_csv(SUMMARY_PATH, index=False)
        
        # LOG LÊN MLFLOW
        # mlflow.log_artifact(SUMMARY_PATH)
        mlflow.log_metric("global_avg_rmse", summary["rmse"].mean())

        print("\n Looking for a Champion...")
        # Gọi file chọn model (file này sẽ tạo 1 run MLflow riêng hoặc có thể lồng vào đây)
        subprocess.run(["python3", "select_best_model_2.py"])
    else:
        print("There are no results to summarize.")