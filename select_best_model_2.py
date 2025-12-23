import pandas as pd
import shutil
import os

# =====================
# CONFIG
# =====================
SUMMARY_PATH = "evaluation_logs/evaluation_summary.csv"
MODEL_SOURCE_DIR = "top3_models_incremental"
BEST_MODEL_DIR = "best_model_final"

def select_the_champion():
    # 1. Kiểm tra file summary có tồn tại không
    if not os.path.exists(SUMMARY_PATH):
        print(f"❌ Không tìm thấy file summary tại: {SUMMARY_PATH}")
        return

    # 2. Đọc file summary
    df = pd.read_csv(SUMMARY_PATH)

    if df.empty:
        print("❌ File summary trống!")
        return

    # 3. Lấy model đứng đầu (vì summary đã được sort_values("rmse") ở bước trước)
    # Nếu chưa sort, có thể dùng: df.loc[df['rmse'].idxmin()]
    best_model_info = df.iloc[0]
    best_model_name = best_model_info['model']
    best_rmse = best_model_info['rmse']

    print(f"🏆 Model tốt nhất xác định được là: {best_model_name}")
    print(f"📉 Chỉ số RMSE trung bình: {best_rmse:.4f}")

    # 4. Tạo thư mục lưu trữ model tốt nhất
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    source_path = os.path.join(MODEL_SOURCE_DIR, best_model_name)
    destination_path = os.path.join(BEST_MODEL_DIR, "weather_model_production.pth")

    # 5. Copy và đổi tên để dễ quản lý trong môi trường Production/Jenkins
    try:
        shutil.copy(source_path, destination_path)
        print(f"✅ Đã copy model vào: {destination_path}")
        
        # Lưu kèm 1 file text ghi chú thông số của model này
        with open(f"{BEST_MODEL_DIR}/model_info.txt", "w") as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Average RMSE: {best_rmse}\n")
            f.write(f"Average MAE: {best_model_info['mae']}\n")
            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file model gốc tại: {source_path}")

if __name__ == "__main__":
    select_the_champion()