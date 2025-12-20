import os
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from s3io import read_csv

warnings.filterwarnings("ignore")

# =========================================================
#      HARDCODED PATHS (DÙNG ĐỂ TEST)
# =========================================================
DATA_PATH = "./weather_dataset.csv"
MODEL_OUTPUT_PATH = "./model.pth"

# =========================
#         CONFIG
# =========================
@dataclass
class CFG:
    timestamp_col: str = "timestamp"
    province_col: str = "province"
    province_filter: Optional[str] = None

    feature: List[str] = None
    target: List[str] = None

    seq_len: int = 30
    horizon: int = 10

    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3

# =========================
#      DATASET
# =========================
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
#      TCN COMPONENTS
# =========================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs, channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(
                TCNBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        return self.out(y[:, :, -1])

# =========================
#         MAIN
# =========================
def main():
    cfg = CFG()
    cfg.feature = [
        "temperature", "feels_like", "humidity", "wind_speed",
        "gust_speed", "pressure", "precipitation",
        "rain_probability", "snow_probability", "uv_index",
        "dewpoint", "visibility", "cloud"
    ]
    cfg.target = cfg.feature.copy()

    target_cols = [f"{c}_t+{cfg.horizon}" for c in cfg.target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading data from {DATA_PATH}")

    df = read_csv(DATA_PATH)

    if cfg.province_filter:
        df = df[df[cfg.province_col] == cfg.province_filter].copy()

    df_feat = df[cfg.feature].copy()
    for c in cfg.target:
        df_feat[f"{c}_t+{cfg.horizon}"] = df[c].shift(-cfg.horizon)

    df_feat.dropna(inplace=True)

    X = df_feat[cfg.feature].values
    y = df_feat[target_cols].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    def create_sequences(X, y, seq_len):
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len - 1])
        return np.array(xs), np.array(ys)

    X_tr, y_tr = create_sequences(X_tr, y_tr, cfg.seq_len)
    X_val, y_val = create_sequences(X_val, y_val, cfg.seq_len)

    train_loader = DataLoader(
        WeatherDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        WeatherDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False
    )

    model = TCN(
        num_inputs=len(cfg.feature),
        num_outputs=len(target_cols),
        channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item()

        print(f"Epoch {ep+1}: val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    torch.save(
        {
            "state_dict": best_state,
            "features": cfg.feature,
            "targets": target_cols,
            "seq_len": cfg.seq_len,
            "horizon": cfg.horizon,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        MODEL_OUTPUT_PATH
    )

    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
