import os
import subprocess
import pandas as pd

TEST_CASES = [
    "weather_test_1.py",
    "weather_test_2.py",
    "weather_test_3.py",
]
LOG_DIR = "test_logs"
EVAL_LOG_DIR = "evaluation_logs"
DETAIL_PATH = os.path.join(EVAL_LOG_DIR, "evaluation_detail.csv")
SUMMARY_PATH = os.path.join(EVAL_LOG_DIR, "evaluation_summary.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

print("🚀 Running all test cases...")
all_results = []

for idx, case in enumerate(TEST_CASES, 1):
    print(f"▶️ Running {case} ...")
    # Chạy từng case
    subprocess.run(["python", case])

    # Đọc kết quả chi tiết từng case (giả sử mỗi file test sẽ lưu file CSV riêng)
    case_csv = os.path.join(LOG_DIR, f"case_{idx}_result.csv")
    if os.path.exists(case_csv):
        df_case = pd.read_csv(case_csv)
        df_case["case"] = case.replace(".py", "")
        all_results.append(df_case)
    else:
        print(f"❌ Could not find file {case_csv} for {case}")

# Gộp tất cả kết quả chi tiết
if all_results:
    df_detail = pd.concat(all_results, ignore_index=True)
    df_detail.to_csv(DETAIL_PATH, index=False)
else:
    print("❌ There are no test results to aggregate.")
    exit(1)


# Tính summary trung bình theo model
summary = (
    df_detail
    .groupby("model")[["mae", "rmse"]]
    .mean()
    .reset_index()
    .sort_values("rmse")
)
summary.to_csv(SUMMARY_PATH, index=False)

print("\n🏆 FINAL SUMMARY")
print(summary.to_string(index=False))