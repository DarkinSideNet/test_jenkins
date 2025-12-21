import pandas as pd
import os
import sys
import shutil
import subprocess

# =====================
# CONFIG
# =====================
RESULTS_PATH = "./evaluation_logs/evaluation_summary.csv"
MODEL_SOURCE_DIR = "./models"
PROD_LOCAL_DIR = "./production_ready"
# PRODUCTION_PATH = "myminio/devopsproject/Production_Model"
PROD_MODEL_NAME = "weather_prod_model.pth"


# =====================
# MAIN
# =====================
def main():
    # -------------------------------------------------
    # 1. Check evaluation result
    # -------------------------------------------------
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ ERROR: {RESULTS_PATH} not found.")
        print("👉 Please run evaluation step first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_PATH)

    required_cols = {"model", "mae", "rmse"}
    if not required_cols.issubset(df.columns):
        print(f"❌ ERROR: evaluation file missing columns {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    # -------------------------------------------------
    # 2. Rank models (average over all test cases)
    # -------------------------------------------------
    summary = (
        df.groupby("model")[["mae", "rmse"]]
        .mean()
        .reset_index()
        .sort_values("mae")
    )

    print("\n📊 MODEL RANKING (AVERAGE OVER ALL TEST CASES)")
    print(summary.to_string(index=False))

    # -------------------------------------------------
    # 3. Select Champion
    # -------------------------------------------------
    winner = summary.iloc[0]
    winner_model_name = winner["model"]
    winner_mae = winner["mae"]
    winner_rmse = winner["rmse"]

    print("\n🏆 CHAMPION MODEL SELECTED")
    print(f"Model : {winner_model_name}")
    print(f"MAE   : {winner_mae:.6f}")
    print(f"RMSE  : {winner_rmse:.6f}")

    # -------------------------------------------------
    # 4. Validate physical model file
    # -------------------------------------------------
    winner_src_path = os.path.join(MODEL_SOURCE_DIR, winner_model_name)
    if not os.path.exists(winner_src_path):
        print(f"❌ ERROR: Model file not found: {winner_src_path}")
        sys.exit(1)

    # -------------------------------------------------
    # 5. Copy to production_ready (local)
    # -------------------------------------------------
    os.makedirs(PROD_LOCAL_DIR, exist_ok=True)
    local_prod_path = os.path.join(PROD_LOCAL_DIR, PROD_MODEL_NAME)

    shutil.copy(winner_src_path, local_prod_path)
    print(f"\n✅ Local production copy created:")
    print(f"   {local_prod_path}")

    # -------------------------------------------------
    # 6. Deploy to MinIO (optional)
    # -------------------------------------------------
    # dest_minio_path = f"{PRODUCTION_PATH}/{PROD_MODEL_NAME}"

    # print(f"\n🚀 Deploying to MinIO:")
    # print(f"   {dest_minio_path}")

    # 👉 Uncomment khi sẵn sàng deploy thật
    # try:
    #     subprocess.run(
    #         ["mcli", "cp", local_prod_path, dest_minio_path],
    #         check=True
    #     )
    #     print("✅ SUCCESS: Model deployed to Production!")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Deployment failed: {e}")
    #     sys.exit(1)


# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    main()
