import os
import subprocess
import pandas as pd

# =====================
# CONFIG
# =====================
# Tự động quét tất cả file .csv trong folder dataset_test
DATASET_DIR = "dataset_test"
TEST_DATASETS = sorted([
    os.path.join(DATASET_DIR, f) 
    for f in os.listdir(DATASET_DIR) 
    if f.endswith(".csv")
])

LOG_DIR = "test_logs"
EVAL_LOG_DIR = "evaluation_logs"
DETAIL_PATH = os.path.join(EVAL_LOG_DIR, "evaluation_detail.csv")
SUMMARY_PATH = os.path.join(EVAL_LOG_DIR, "evaluation_summary.csv")

# Tạo thư mục nếu chưa có
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

# =====================
# 1. CHẠY TEST CASES
# =====================
print(f"🚀 Found {len(TEST_DATASETS)} datasets in {DATASET_DIR}. Starting tests...")

for idx, data_path in enumerate(TEST_DATASETS, 1):
    case_name = f"case_{idx}"
    print(f"\n▶️ Running {case_name} | Data: {os.path.basename(data_path)} ...")
    
    # Gọi file test duy nhất kèm tham số đầu vào
    # --data: đường dẫn file csv
    # --out_name: tên file kết quả (case_1, case_2...)
    subprocess.run([
        "python3", "weather_test.py", 
        "--data", data_path, 
        "--out_name", case_name
    ])

# =====================
# 2. TỔNG HỢP KẾT QUẢ
# =====================
print("\n📊 Aggregating results...")
all_results = []

# Quét lại folder test_logs để tìm các file case_X_result.csv vừa tạo ra
for idx in range(1, len(TEST_DATASETS) + 1):
    case_csv = os.path.join(LOG_DIR, f"case_{idx}_result.csv")
    
    if os.path.exists(case_csv):
        df_case = pd.read_csv(case_csv)
        # Thêm cột để biết kết quả này từ dataset nào
        df_case["test_dataset"] = os.path.basename(TEST_DATASETS[idx-1])
        all_results.append(df_case)
    else:
        print(f"❌ Warning: Result file {case_csv} not found.")

if not all_results:
    print("❌ No results found. Evaluation failed.")
    exit(1)

# Gộp tất cả chi tiết
df_detail = pd.concat(all_results, ignore_index=True)
df_detail.to_csv(DETAIL_PATH, index=False)
print(f"✅ Saved detailed results to: {DETAIL_PATH}")

# =====================
# 3. TÍNH TOÁN SUMMARY
# =====================
# Tính trung bình MAE và RMSE của mỗi model dựa trên tất cả các test case
summary = (
    df_detail
    .groupby("model")[["mae", "rmse"]]
    .mean()
    .reset_index()
    .sort_values("rmse")
)
summary.to_csv(SUMMARY_PATH, index=False)

print("\n🏆 FINAL SUMMARY (Averaged across all test datasets)")
print("-" * 60)
print(summary.to_string(index=False))
print("-" * 60)
print(f"✅ Summary saved to: {SUMMARY_PATH}")