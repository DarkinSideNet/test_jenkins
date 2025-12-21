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
TEST_DATA_PATH = "dataset_test/weather_2024-06-15_to_2024-06-16.csv"

TOP3_MODELS = [
    "h6_ep50_bs64.pth",
    "h12_ep50_bs64.pth",
    "h12_ep30_bs64.pth",
]

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

    X_seq, y_seq = create_sequences(X, y, seq_len)
    return X_seq, y_seq


# =====================
# TEST ONE MODEL
# =====================
def test_one_model(model_path, df_test):
    print(f"\nTesting model: {model_path}")

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
            f"Not enough test samples for model {model_path}. "
            f"Need >= seq_len + horizon + 1 rows."
        )

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = TCN(len(features), len(targets))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(X_test).numpy()

    y_true = y_test.numpy()

    # =====================
    # SAVE DETAIL CSV 
    # =====================
    out_data = {}
    for i, t in enumerate(targets):
        out_data[f"{t}_true"] = y_true[:, i]
        out_data[f"{t}_pred"] = preds[:, i]

    os.makedirs("test_logs", exist_ok=True)
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(
        os.path.join(
            "test_logs",
            f"detail_{os.path.basename(model_path).replace('.pth', '')}.csv"
        ),
        index=False
    )

    # =====================
    # METRICS
    # =====================
    mae_total = mean_absolute_error(y_true, preds)
    rmse_total = np.sqrt(mean_squared_error(y_true, preds))

    mae_per_target = {}
    rmse_per_target = {}

    for i, t in enumerate(targets):
        mae_per_target[t] = mean_absolute_error(y_true[:, i], preds[:, i])
        rmse_per_target[t] = np.sqrt(
            mean_squared_error(y_true[:, i], preds[:, i])
        )

    print(f"   Horizon: {horizon}")
    print(f"   MAE (overall): {mae_total:.4f}")
    print(f"   RMSE (overall): {rmse_total:.4f}")

    return {
        "model": os.path.basename(model_path),
        "horizon": horizon,
        "mae": mae_total,
        "rmse": rmse_total,
        "mae_per_target": mae_per_target,
        "rmse_per_target": rmse_per_target,
    }


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    df_test = pd.read_csv(TEST_DATA_PATH)

    results = []
    log_lines = []

    for model_name in TOP3_MODELS:
        model_path = os.path.join(MODEL_DIR, model_name)
        result = test_one_model(model_path, df_test)
        results.append(result)

        log_lines.append(
            f"Model: {result['model']} | "
            f"Horizon: {result['horizon']} | "
            f"MAE: {result['mae']:.4f} | "
            f"RMSE: {result['rmse']:.4f}"
        )

    # =====================
    # FINAL COMPARISON
    # =====================
    df_results = pd.DataFrame(
        [
            {
                "model": r["model"],
                "horizon": r["horizon"],
                "mae": r["mae"],
                "rmse": r["rmse"],
            }
            for r in results
        ]
    )

    log_lines.append("\nFINAL COMPARISON")
    log_lines.append(df_results.sort_values("rmse").to_string(index=False))

    os.makedirs("test_logs", exist_ok=True)
    with open(os.path.join("test_logs", "test_results.log"), "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print("\n🏆 FINAL COMPARISON")
    print(df_results.sort_values("rmse"))
