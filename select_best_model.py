import pandas as pd
import subprocess
import os
import sys
import shutil

# --- CONFIG ---
RESULTS_PATH = "./evaluation_results.csv"
MODEL_SOURCE_DIR = "./best_models"
PROD_LOCAL_DIR = "./production_ready"
# Đường dẫn đích trên MinIO cho mô hình sẵn sàng chạy thực tế
PRODUCTION_PATH = "myminio/devopsproject/Production_Model/"
# Tên file thống nhất để API (FastAPI) có thể load cố định
PROD_MODEL_NAME = "weather_prod_model.pth"

def main():
    print("\n" + "="*50)
    print("STEP 4: SELECTING THE BEST MODEL FOR PRODUCTION")
    print("="*50)

    # 1. Kiểm tra file kết quả
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found! Please run Phase 2 first.")
        sys.exit(1)

    # 2. Đọc dữ liệu kết quả đánh giá
    df = pd.read_csv(RESULTS_PATH)
    
    # 3. Tính toán Ranking dựa trên Average MAE
    # Group by theo Model và tính trung bình MAE trên tất cả các test datasets
    summary = df.groupby("Model")["MAE"].mean().reset_index()
    
    # 4. Tìm Model có MAE thấp nhất (The Champion)
    winner_row = summary.loc[summary['MAE'].idxmin()]
    winner_model_name = winner_row['Model']
    winner_mae = winner_row['MAE']
    
    print(f"The Champion Model is: {winner_model_name}")
    print(f"Average MAE across all tests: {winner_mae:.6f}")
    print("-" * 50)

    # 5. Kiểm tra file vật lý của Champion
    winner_file_path = os.path.join(MODEL_SOURCE_DIR, winner_model_name)
    if not os.path.exists(winner_file_path):
        print(f" Error: Physical file {winner_file_path} not found!")
        sys.exit(1)

    os.makedirs(PROD_LOCAL_DIR, exist_ok=True)
    local_dest_path = os.path.join(PROD_LOCAL_DIR, PROD_MODEL_NAME)

    shutil.copy(winner_file_path, local_dest_path)
    print(f"Local copy successful: {local_dest_path}")

    # 6. Triển khai (Deploy) bằng MCLI
    # Chúng ta sẽ copy và đổi tên thành 'weather_prod_model.pth' 
    # để các hệ thống phía sau (như FastAPI) luôn gọi đúng 1 tên file duy nhất.
    dest_full_path = os.path.join(PRODUCTION_PATH, PROD_MODEL_NAME)
    
    print(f"Deploying to Production: {dest_full_path} ...")
    
    # try:
    #     # Lệnh: mcli cp ./best_models/top_X.pth myminio/devopsproject/Production_Model/weather_prod_model.pth
    #     subprocess.run(["mcli", "cp", winner_file_path, dest_full_path], check=True)
        
    #     print(f"SUCCESS: Model is now live in Production!")
    #     print(f"Timestamp: {pd.Timestamp.now()}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Deployment failed during MCLI execution: {e}")
    #     sys.exit(1)

if __name__ == "__main__":
    main()