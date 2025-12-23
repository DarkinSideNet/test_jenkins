import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================
# CẤU HÌNH ĐỘNG
# =====================
MODEL_DIR = "top3_models_incremental" # Tự động lấy model mới nhất
LOG_DIR = "test_logs"

TARGETS = [
    "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
    "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
]

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

def prepare_data(df, features, targets, horizon, seq_len, scaler_X):
    df = df.copy()
    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)
    df.dropna(inplace=True)
    # Dùng .values để tránh cảnh báo UserWarning về feature names
    X = scaler_X.transform(df[features].values)
    y = df[[f"{t}_y" for t in targets]].values
    
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)

def test_one_model(model_path, df_test):
    # Load model và các thông số đi kèm
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Lấy config từ file train (Lưu ý: dùng đúng key 'config')
    cfg = checkpoint["config"]
    features = checkpoint["features"]
    targets = checkpoint["targets"]
    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]
    if "scaler_X" in checkpoint:
        scaler_X = checkpoint["scaler_X"]
    else:
        # Nếu model lưu kiểu cũ (mean/scale rời rạc), ta tự tạo lại scaler
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(checkpoint["scaler_mean"])
        scaler_X.scale_ = np.array(checkpoint["scaler_scale"])
        scaler_X.n_features_in_ = len(features)
    X_test, y_test = prepare_data(df_test, features, targets, horizon, seq_len, scaler_X)
    
    if len(X_test) == 0:
        return None, None, None, horizon

    model = TCN(len(features), len(targets))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_t).numpy()
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    out_detail = {}
    for i, t in enumerate(targets):
        out_detail[f"{t}_true"] = y_test[:, i]
        out_detail[f"{t}_pred"] = preds[:, i]
        
    return out_detail, mae, rmse, horizon

if __name__ == "__main__":
    # Thiết lập nhận tham số từ command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file CSV test")
    parser.add_argument("--out_name", type=str, required=True, help="Tên file log đầu ra (ví dụ: case_1)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    df_test = pd.read_csv(args.data)
    
    # Tự động quét các model trong folder incremental
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    
    all_rows = []
    for m_name in model_files:
        path = os.path.join(MODEL_DIR, m_name)
        detail, mae, rmse, horizon = test_one_model(path, df_test)
        
        if detail:
            n_samples = len(next(iter(detail.values())))
            for i in range(n_samples):
                row = {"model": m_name, "horizon": horizon, "mae": mae, "rmse": rmse}
                for k, v in detail.items():
                    row[k] = v[i]
                all_rows.append(row)

    if all_rows:
        result_df = pd.DataFrame(all_rows)
        out_path = os.path.join(LOG_DIR, f"{args.out_name}_result.csv")
        result_df.to_csv(out_path, index=False)
        print(f"Testing completed for {len(model_files)} models. Results saved at: {out_path}")