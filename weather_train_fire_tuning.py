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

DATA_PATH = "./dataset.csv"
MODEL_OUTPUT_PATH = "./model.pth"

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
    timestamp_col: str = "timestamp"
    province_col: str = "province"
    province_filter: Optional[str] = None
    feature: List[str] = None
    target: List[str] = None
    seq_len: int = 30
    horizon: int = 10
    epochs: int = 20
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
def main():


    parser = argparse.ArgumentParser(description="TCN Weather Forecasting Training Script")
    parser.add_argument('--data_path', type=str, default="./weather_dataset.csv", help='Path to data')
    parser.add_argument('--output_path', type=str, default="./model.pth", help='Path to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon')
    args = parser.parse_args()
    cfg = CFG()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.horizon = args.horizon
    DATA_PATH = args.data_path
    MODEL_OUTPUT_PATH = args.output_path
    cfg = CFG()
    cfg.feature = [
        "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
        "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
    ]
    cfg.target = [
        "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
        "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
    ]


    cfg.target = cfg.feature
    target_cols = [f"{c}_t+{cfg.horizon}" for c in cfg.target]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # target_cols = [f"{c}_t+{cfg.horizon}" for c in cfg.target]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # # Load data from hardcoded path
    # print(f"Loading data from {DATA_PATH}...")
    # df = read_csv(DATA_PATH)
    
    # # Process data
    # # Use fixed 'timestamp' column without sorting (data already ordered at collection time)
    ts_col = cfg.timestamp_col
    # if cfg.province_filter is not None:
    #     df = df[df[cfg.province_col] == cfg.province_filter].copy()
    # df_feat = df[cfg.feature].copy()
    # for c in cfg.target:
    #     df_feat[f"{c}_t+{cfg.horizon}"] = df[c].shift(-cfg.horizon)
    # df_feat = df_feat.dropna().reset_index(drop=True)
    
    # # Split & Scale
    # X = df_feat[cfg.feature].values
    # y = df_feat[target_cols].values
    # X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # scaler = StandardScaler()
    # X_tr_scaled = scaler.fit_transform(X_tr)
    # X_val_scaled = scaler.transform(X_val)
    
    # Create sequences
    # def create_sequences(X, y, seq_len):
    #     Xs, ys = [], []
    #     for i in range(len(X) - seq_len):
    #         Xs.append(X[i:i+seq_len])
    #         ys.append(y[i+seq_len-1])
    #     return np.array(Xs), np.array(ys)
    # X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr, cfg.seq_len)
    # X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, cfg.seq_len)

    # # Dataloader
    # train_ds = WeatherDataset(X_tr_seq, y_tr_seq)
    # val_ds = WeatherDataset(X_val_seq, y_val_seq)
    # train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # Model, Loss, Optimizer
    model = TCN(
        num_inputs=len(cfg.feature),
        num_outputs=len(target_cols),
        channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2,
    ).to(device)

    scaler = StandardScaler()
    is_fine_tuning = False

    if os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Found existing model at {MODEL_OUTPUT_PATH}. Loading for fine-tuning..")
        checkpoint = torch.load(MODEL_OUTPUT_PATH, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])
        if 'scaler_mean' in checkpoint:
            scaler.mean_ = np.array(checkpoint['scaler_mean'])
            scaler.scale_ = np.array(checkpoint['scaler_scale'])
            scaler.n_features_in_ = len(cfg.feature)
            is_fine_tuning = True
        cfg.lr = cfg.lr * 0.1
        print(f"Fine-tuning mode: Learning rate adjusted to {cfg.lr}")

    df = pd.read_csv(DATA_PATH)

    if cfg.province_filter is not None:
        df = df[df[cfg.province_col] == cfg.province_filter].copy()

    df_feat = df[cfg.feature].copy()
    for c in cfg.target:
        df_feat[f"{c}_t+{cfg.horizon}"] = df[c].shift(-cfg.horizon)
    df_feat = df_feat.dropna().reset_index(drop=True)

    X = df_feat[cfg.feature].values
    y = df_feat[target_cols].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    if is_fine_tuning:
        X_tr_scaled = scaler.transform(X_tr)
    else:
        X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)


    # [Tạo Sequences và Dataloader - Giữ nguyên logic của bạn]
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len-1])
        return np.array(Xs), np.array(ys)
    
    X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr, cfg.seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, cfg.seq_len)

    train_ds = WeatherDataset(X_tr_seq, y_tr_seq)
    val_ds = WeatherDataset(X_val_seq, y_val_seq)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    #training setup
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    best_state = model.state_dict()
    for ep in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        
        # Validation logic...
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(yb)
        
        v_l = val_loss / len(val_ds)
        print(f"Epoch {ep+1}: train {tr_loss/len(train_ds):.4f} | val {v_l:.4f}")
        
        if v_l < best_val:
            best_val = v_l
            best_state = model.state_dict()
    # checkpoint_to_save = {
    #     "state_dict": best_state,
    #     "features": cfg.feature,
    #     "targets": target_cols,
    #     "seq_len": cfg.seq_len,
    #     "horizon": cfg.horizon,
    #     "scaler_mean": scaler.mean_.tolist(),
    #     "scaler_scale": scaler.scale_.tolist(),
    # }
    # torch.save(checkpoint_to_save, MODEL_OUTPUT_PATH)
    print(f"--> Model successfully saved/updated at: {MODEL_OUTPUT_PATH}")

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
        "val_loss": best_val
    }, MODEL_OUTPUT_PATH)


    print("--> Training and saving completed successfully.")
if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Train a TCN model for weather forecasting.")
    # parser.add_argument('--data-path', type=str, required=True, help='Path to the weather.csv data file.')
    # parser.add_argument('--model-output-path', type=str, required=True, help='Path to save the output model.pth file.')
    
    # # Phân tích các đối số từ dòng lệnh
    # args = parser.parse_args()
    main()
