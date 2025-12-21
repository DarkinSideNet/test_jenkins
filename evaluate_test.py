import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================
# TCN MODEL
# =====================
class TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_inputs, 32, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=4, dilation=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.net(x)
        return self.fc(y[:, :, -1])


# =====================
# CONFIG
# =====================
MODEL_DIR = "models"

# 🔴 3 FILE TEST – 3 KỊCH BẢN
TEST_DATASETS = {
    "case_1": "dataset_test/weather_2025-01-01_to_2025-01-02.csv",
    "case_2": "dataset_test/weather_2024-06-15_to_2024-06-16.csv",
    "case_3": "dataset_test/weather_2024-12-15_to_2024-12-16.csv",
}

TOP3_MODELS = [
    "h6_ep50_bs64.pth",
    "h12_ep50_bs64.pth",
    "h12_ep30_bs64.pth",
]

OUT_DIR = "evaluation_logs"


# =====================
# UTILS
# =====================
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)


def prepare_test_data(df, features, targets, horizon, seq_len, scaler_X):
    df = df.copy()

    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)

    df.dropna(inplace=True)

    X = scaler_X.transform(df[features])
    y = df[[f"{t}_y" for t in targets]].values

    return create_sequences(X, y, seq_len)


# =====================
# TEST ONE MODEL ON ONE DATASET
# =====================
def test_one_model_on_dataset(model_path, df_test, dataset_name):
    checkpoint = torch.load(model_path, weights_only=False)

    features = checkpoint["features"]
    targets = checkpoint["targets"]
    scaler_X = checkpoint["scaler_X"]
    cfg = checkpoint["config"]

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]

    X_test, y_test = prepare_test_data(
        df_test, features, targets, horizon, seq_len, scaler_X
    )

    if len(X_test) == 0:
        raise ValueError(
            f"Dataset {dataset_name} too short for model {model_path} "
            f"(need >= seq_len + horizon + 1 rows)"
        )

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = TCN(len(features), len(targets))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(X_test_t).numpy()

    y_true = y_test_t.numpy()

    # =====================
    # SAVE DETAIL CSV (PER TEST CASE)
    # =====================
    os.makedirs("test_logs", exist_ok=True)
    case_dir = os.path.join("test_logs", dataset_name)
    os.makedirs(case_dir, exist_ok=True)

    out_data = {}
    for i, t in enumerate(targets):
        out_data[f"{t}_true"] = y_true[:, i]
        out_data[f"{t}_pred"] = preds[:, i]

    detail_path = os.path.join(
        case_dir,
        f"detail_{os.path.basename(model_path).replace('.pth', '')}.csv"
    )
    pd.DataFrame(out_data).to_csv(detail_path, index=False)

    # =====================
    # METRICS
    # =====================
    mae_total = mean_absolute_error(y_true, preds)
    rmse_total = np.sqrt(mean_squared_error(y_true, preds))

    return {
        "model": os.path.basename(model_path),
        "dataset": dataset_name,
        "horizon": horizon,
        "mae": mae_total,
        "rmse": rmse_total,
    }


# =====================
# MAIN EVALUATION
# =====================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    log_lines = []

    print(f"🚀 Evaluating {len(TOP3_MODELS)} models on {len(TEST_DATASETS)} datasets")

    for dataset_name, dataset_path in TEST_DATASETS.items():
        print(f"\nLoading dataset: {dataset_name}")
        df_test = pd.read_csv(dataset_path)

        for model_name in TOP3_MODELS:
            model_path = os.path.join(MODEL_DIR, model_name)
            print(f"   Model: {model_name}")

            try:
                result = test_one_model_on_dataset(
                    model_path, df_test, dataset_name
                )
                all_results.append(result)

                line = (
                    f"Dataset={dataset_name} | "
                    f"Model={result['model']} | "
                    f"Horizon={result['horizon']} | "
                    f"MAE={result['mae']:.4f} | "
                    f"RMSE={result['rmse']:.4f}"
                )
                print("     ✅", line)
                log_lines.append(line)

            except Exception as e:
                err = (
                    f"Dataset={dataset_name} | "
                    f"Model={model_name} | ERROR: {e}"
                )
                print("     ❌", err)
                log_lines.append(err)

    # =====================
    # FINAL SUMMARY
    # =====================
    df_results = pd.DataFrame(all_results)

    summary = (
        df_results
        .groupby("model")[["mae", "rmse"]]
        .mean()
        .reset_index()
        .sort_values("rmse")
    )

    print("\n🏆 FINAL EVALUATION SUMMARY")
    print(summary.to_string(index=False))

    df_results.to_csv(
        os.path.join(OUT_DIR, "evaluation_detail.csv"),
        index=False
    )
    summary.to_csv(
        os.path.join(OUT_DIR, "evaluation_summary.csv"),
        index=False
    )

    with open(os.path.join(OUT_DIR, "evaluation.log"), "w", encoding="utf-8") as f:
        for l in log_lines:
            f.write(l + "\n")
