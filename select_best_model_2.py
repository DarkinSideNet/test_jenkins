import pandas as pd
import shutil
import os
import mlflow
# =====================
# CONFIG
# =====================
SUMMARY_PATH = "evaluation_logs/evaluation_summary.csv"
MODEL_SOURCE_DIR = "top3_models_incremental"
BEST_MODEL_DIR = "best_model_final"

def select_the_champion():
    if not os.path.exists(SUMMARY_PATH): return
    df = pd.read_csv(SUMMARY_PATH)
    if df.empty: return

    best_model_info = df.iloc[0]
    best_model_name = best_model_info['model']
    best_rmse = best_model_info['rmse']

    # ĐỊNH NGHĨA ĐƯỜNG DẪN TRƯỚC KHI LOG
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    source_path = os.path.join(MODEL_SOURCE_DIR, best_model_name)
    destination_path = os.path.join(BEST_MODEL_DIR, "weather_model_production.pth")
    info_path = os.path.join(BEST_MODEL_DIR, "model_info.txt")

    # Lưu file info cục bộ trước
    with open(info_path, "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"RMSE: {best_rmse:.4f}")

    # Thực hiện copy model cục bộ
    shutil.copy(source_path, destination_path)

    # LOG LÊN MLFLOW
    mlflow.set_tracking_uri("https://mlflow.neikoscloud.net")
    mlflow.set_experiment("weather_evaluation") # Hoặc tạo exp mới "champion_selection"
    
    with mlflow.start_run(run_name="Champion_Final"):
        mlflow.log_param("champion_model", best_model_name)
        mlflow.log_metric("best_rmse", best_rmse)
        # Log cả model và file info lên server
        mlflow.log_artifact(info_path)
        mlflow.log_artifact(destination_path, artifact_path="production_ready")

    print(f"✅ Đã chọn Champion: {best_model_name} với RMSE: {best_rmse:.4f}")
if __name__ == "__main__":
    select_the_champion()