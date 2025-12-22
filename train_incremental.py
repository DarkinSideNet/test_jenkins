import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

BEST_MODEL_PATH = "./production_ready/weather_prod_model.pth"
FINE_TUNED_MODEL_PATH = "./production_ready/weather_prod_model_finetuned.pth"

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

# =========================
#      FINE-TUNE LOGIC
# =========================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)

def prepare_data(df, features, targets, horizon, seq_len, scaler_X):
    df = df.copy()
    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)
    df.dropna(inplace=True)
    X = scaler_X.transform(df[features])
    y = df[[f"{t}_y" for t in targets]].values
    X_seq, y_seq = create_sequences(X, y, seq_len)
    return X_seq, y_seq

def finetune_on_new_data(new_csv_path, epochs=5, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load best model checkpoint
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
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

    # 2. Rebuild model
    model = TCN(len(features), len(targets))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # 3. Load and process new data
    df = pd.read_csv(new_csv_path)
    X_seq, y_seq = prepare_data(df, features, targets, horizon, seq_len, scaler_X)
    if len(X_seq) == 0:
        print("Not enough data for fine-tuning (need at least seq_len + horizon rows).")
        return

    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_seq, y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fine-tune
    model.train()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
        print(f"Epoch {ep+1}/{epochs}: Fine-tune loss = {total_loss/len(dataset):.4f}")

    # Save fine-tuned model
    torch.save({
        "state_dict": model.state_dict(),
        "features": features,
        "targets": targets,
        "seq_len": seq_len,
        "horizon": horizon,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "config": checkpoint.get("config", {}),
    }, FINE_TUNED_MODEL_PATH)
    print(f"✅ Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")


# =========================
#      ENTRY POINT
# =========================
if __name__ == "__main__":
    daily_dir = "./dataset_daily/"
    daily_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    if not daily_files:
        print("No daily dataset found for fine-tuning.")
        exit(1)
    daily_csv = daily_files[-1]
    print(f"Using daily dataset for fine-tuning: {daily_csv}")
    
    epochs = 20
    batch_size = 64
    lr = 1e-3
    
    finetune_on_new_data(daily_csv, epochs=epochs, batch_size=batch_size, lr=lr)