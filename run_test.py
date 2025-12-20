import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

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
        for i in range(len(channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i-1]
            out_channels = channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, 1, dilation, (kernel_size-1)*dilation, dropout)]
        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], num_outputs)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        o = self.network(x)
        o = self.out(o[:, :, -1])
        return o

def load_ckpt(pt_path="model.pth"):
    """
    Load checkpoint + dựng đúng kiến trúc TCN với số output = số target.
    """
    ckpt = torch.load(pt_path, map_location="cpu")

    # Lấy danh sách target
    if "targets" in ckpt and isinstance(ckpt["targets"], (list, tuple)):
        targets = list(ckpt["targets"])
    elif "target" in ckpt:
        targets = [ckpt["target"]]
    else:
        raise KeyError("Checkpoint không chứa 'targets' hoặc 'target'.")
    num_inputs = len(ckpt["features"])
    num_outputs = len(targets)
    
    # Dựng model với đúng số kênh input và số output
    model = TCN(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        channels=[32, 64, 128], # Thông số này bạn dùng cố định lúc train
        kernel_size=3,          # Thông số này bạn dùng cố định lúc train
        dropout=0.2
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    #dựng sacler từ mean/scale đã lưu (sklearn cần n_features_in)
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = scaler.mean_.shape[0]
    try:
        scaler.feature_names_in_ = np.array(ckpt["features"], dtype=object)
    except Exception:
        pass
    return model, scaler, ckpt, targets

def predict_with_new_data(df_new: pd.DataFrame, model, scaler, ckpt, targets):
    """ df_new: DataFrame mới thu thập (chứa ÍT NHẤT các cột feature giống lúc train).
    Trả về: dict chứa dự báo cho MỤC TIÊU (multi-target), cùng horizon/seq_len để tham chiếu."""
    feat_cols = ckpt["features"]
    seq_len   = int(ckpt["seq_len"])

    # Làm sạch tối thiểu theo thời gian (nếu có)
    df_new = df_new.copy()
    ts_col = "date" if "date" in df_new.columns else ("timestamp" if "timestamp" in df_new.columns else None)
    if ts_col:
        df_new[ts_col] = pd.to_datetime(df_new[ts_col], errors="coerce")
        df_new = df_new.sort_values(ts_col)
    
    # Chỉ lấy đúng cột feature theo đúng thứ tự
    miss = [c for c in feat_cols if c not in df_new.columns]
    if miss:
        raise ValueError(f"Dữ liệu mới thiếu cột feature: {miss}")

    X_full = df_new[feat_cols].copy()

    # Xử lý thiếu: forwall fill -> backward fill

    X_full = X_full.ffill().bfill()

    if len(X_full) < seq_len:
        need = seq_len - len(X_full)
        raise ValueError(f"Chưa đủ dữ liệu để dự báo (thiếu {need} bản ghi). Cần >= {seq_len} dòng.")
    # Lấy seq_len bước gần nhất và scale
    last_seq = X_full.tail(seq_len).to_numpy()
    last_seq_scaled = scaler.transform(last_seq).astype(np.float32)

    # Tensor [1, C, T]
    #x = torch.from_numpy(last_seq_scaled).unsqueeze(0).transpose(1, 2)
    x = torch.from_numpy(last_seq_scaled).unsqueeze(0)
    with torch.no_grad():
        y_vec = model(x).cpu().numpy()[0]  # shape (M,)

    # Map kết quả theo tên target
    pred_map = {t: float(v) for t, v in zip(targets, y_vec)}

    return {
        "prediction": pred_map,            # ví dụ {'humidi': ..., 'rain': ...}
        "targets": targets,                # thứ tự target trong ckpt
        "horizon": int(ckpt["horizon"]),
        "seq_len": seq_len,
    }

def main():
    model, scaler, ckpt, targets = load_ckpt("model.pth")
    df_new = pd.read_csv("weather_dataset.csv") 
    res = predict_with_new_data(df_new, model, scaler, ckpt, targets)
    
    preds = res["prediction"]
    print("\n" + "="*30)
    print(f"DỰ BÁO CHO {res['horizon']} BƯỚC TIẾP THEO:")
    print("="*30)
    
    # In ra tất cả những gì có trong preds để tránh sót
    for name, value in preds.items():
        print(f"{name:25}: {value:.3f}")

if __name__ == "__main__":
    main()
