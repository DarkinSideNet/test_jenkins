import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = "models"
MODEL_NAMES = [
    "h6_ep50_bs64.pth",
    "h12_ep50_bs64.pth",
    "h6_ep30_bs64.pth",
]
DATASET_PATH = "dataset_test/weather_2024-06-15_to_2024-06-16.csv"
TARGETS = [
    "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
    "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
]

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)

def prepare_data(df, features, targets, horizon, seq_len, scaler_X):
    df = df.copy()
    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)
    df.dropna(inplace=True)
    X = scaler_X.transform(df[features])
    y = df[[f"{t}_y" for t in targets]].values
    return create_sequences(X, y, seq_len)

class TCN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(num_inputs, 32, 3, padding=2, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 3, padding=4, dilation=2),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(64, num_outputs)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.net(x)
        return self.fc(y[:, :, -1])

def test_one_model(model_path, df_test):
    checkpoint = torch.load(model_path, weights_only=False)
    features = checkpoint["features"]
    targets = checkpoint["targets"]
    seq_len = checkpoint["seq_len"]
    horizon = checkpoint["horizon"]
    scaler_mean = np.array(checkpoint["scaler_mean"])
    scaler_scale = np.array(checkpoint["scaler_scale"])
    scaler_X = StandardScaler()
    scaler_X.mean_ = scaler_mean
    scaler_X.scale_ = scaler_scale
    scaler_X.n_features_in_ = len(features)

    X_test, y_test = prepare_data(df_test, features, targets, horizon, seq_len, scaler_X)
    if len(X_test) == 0:
        raise ValueError("Not enough test data for this model.")

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    model = TCN(len(features), len(targets))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    y_true = y_test_t.numpy()

    # Save detail
    out_data = {}
    for i, t in enumerate(targets):
        out_data[f"{t}_true"] = y_true[:, i]
        out_data[f"{t}_pred"] = preds[:, i]
    return out_data, mean_absolute_error(y_true, preds), np.sqrt(mean_squared_error(y_true, preds)), horizon

def main():
    df_test = pd.read_csv(DATASET_PATH)
    results = []
    for model_name in MODEL_NAMES:
        model_path = f"{MODEL_DIR}/{model_name}"
        try:
            out_data, mae, rmse, horizon = test_one_model(model_path, df_test)
            results.append({
                "model": model_name,
                "horizon": horizon,
                "mae": mae,
                "rmse": rmse,
                "detail": out_data
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "error": str(e)
            })
    return results

if __name__ == "__main__":
    res = main()

    rows = []
    for r in res:
        if "detail" in r and isinstance(r["detail"], dict):
            n = len(next(iter(r["detail"].values())))
            for i in range(n):
                row = {
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "mae": r["mae"],
                    "rmse": r["rmse"],
                }
                for k, v in r["detail"].items():
                    row[k] = v[i]
                rows.append(row)
        else:
            rows.append({
                "model": r.get("model"),
                "error": r.get("error")
            })

    df = pd.DataFrame(rows)
    df.to_csv("test_logs/case_2_result.csv", index=False)
    print("✅ Saved test_logs/case_2_result.csv")