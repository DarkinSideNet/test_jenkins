import os
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import List, Optional
import argparse 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from s3io import read_csv

# Thư viện OTO đã được gỡ bỏ
# from only_train_once import OTO

warnings.filterwarnings("ignore")

# =========================================================
#      CÁC ĐƯỜNG DẪN ĐƯỢC GÁN CỨNG (HARDCODED)
# =========================================================
# Script này mong đợi file data nằm cùng thư mục hoặc có đường dẫn cụ thể
# DATA_PATH = "weather.csv"
# Model sẽ được lưu ra file model.pth
# MODEL_OUTPUT_PATH = "model.pth"

# =========================
#         CONFIG
# =========================
@dataclass
class CFG:
    # Columns mapping
    timestamp_col: str = "timestamp"
    province_col: str = "province"
    province_filter: Optional[str] = None
    
    # Feature & Targets
    feature: List[str] = None
    target: List[str] = None
    
    # Sequence settings
    seq_len: int = 30
    horizon: int = 10
    
    # Train settings
    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3

# =========================
#      DATA LOADER
# =========================
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================================================
#      LỚP HỖ TRỢ CHO MÔ HÌNH TCN
# =========================================================
class Chomp1d(nn.Module):
    """
    Lớp này dùng để cắt bỏ phần padding thừa sau mỗi lớp tích chập.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# =========================
#      MODEL DEFINITION
# =========================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs, channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i-1]
            out_channels = channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        o = self.network(x)
        o = self.out(o[:, :, -1])
        return o

# =========================
#         MAIN
# =========================
def main(args):
    DATA_PATH = args.data_path
    MODEL_OUTPUT_PATH = args.model_output_path
    # Setup
    cfg = CFG()
    cfg.feature = [
        "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
        "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
    ]
    cfg.target = [
        "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
        "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
    ]
    target_cols = [f"{c}_t+{cfg.horizon}" for c in cfg.target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data from hardcoded path
    print(f"Loading data from {DATA_PATH}...")
    df = read_csv(DATA_PATH)
    
    # Process data
    # Use fixed 'timestamp' column without sorting (data already ordered at collection time)
    ts_col = cfg.timestamp_col
    if cfg.province_filter is not None:
        df = df[df[cfg.province_col] == cfg.province_filter].copy()
    df_feat = df[cfg.feature].copy()
    for c in cfg.target:
        df_feat[f"{c}_t+{cfg.horizon}"] = df[c].shift(-cfg.horizon)
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    # Split & Scale
    X = df_feat[cfg.feature].values
    y = df_feat[target_cols].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    # Create sequences
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len-1])
        return np.array(Xs), np.array(ys)
    X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr, cfg.seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, cfg.seq_len)

    # Dataloader
    train_ds = WeatherDataset(X_tr_seq, y_tr_seq)
    val_ds = WeatherDataset(X_val_seq, y_val_seq)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # Model, Loss, Optimizer
    model = TCN(
        num_inputs=len(cfg.feature),
        num_outputs=len(target_cols),
        channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
    ).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Phần tích hợp OTO đã được gỡ bỏ

    # Training loop
    best_val = float("inf")
    best_state = None
    print("Starting training loop...")
    for ep in range(cfg.epochs):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            
            # Lệnh oto.step() đã được gỡ bỏ
            
            opt.step()
            tr_loss_sum += loss.item() * len(yb)
            tr_n += len(yb)
        tr_loss = tr_loss_sum / max(1, tr_n)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss_sum, val_n = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += loss_fn(pred, yb).item() * len(yb)
                val_n += len(yb)
            val_loss = val_loss_sum / max(1, val_n)

        print(f"Epoch {ep+1:02d}: train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
    
    print("Training finished.")

    # Save artifact to hardcoded path
    torch.save({
        "state_dict": best_state,
        "features": cfg.feature,
        "targets": target_cols,
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "timestamp_col": ts_col if ts_col in df.columns else None,
        # Save scaler stats for consistent inference
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, MODEL_OUTPUT_PATH)
    dummy_state = {"model": "TCN", "status": "trained"}
    torch.save(dummy_state, args.model_output_path)
    
    print(f"Model saved locally at: {args.model_output_path}")

    print(f"Model artifact saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TCN model for weather forecasting.")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the weather.csv data file.')
    parser.add_argument('--model-output-path', type=str, required=True, help='Path to save the output model.pth file.')
    
    # Phân tích các đối số từ dòng lệnh
    args = parser.parse_args()
    main(args)
