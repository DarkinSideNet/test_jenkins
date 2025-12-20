import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm

LOCAL_DATA_DIR = "./evaluation_data"
MODEL_DIR = "./best_models"
RESULTS_PATH = "./evaluation_results.csv"

# [Copy các class WeatherDataset, TCNBlock, TCN từ train.py vào đây để script có thể rebuild model]


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



def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)
def evaluate_model(model_path, data_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # 1. Rebuild model dựa trên kiến trúc đã lưu
    model = TCN(num_inputs=len(checkpoint['features']), 
                num_outputs=len(checkpoint['targets']), 
                channels=[32, 64, 128], kernel_size=3, dropout=0.2).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. Đọc dữ liệu
    df = pd.read_csv(data_path)
    features = checkpoint['features']
    target_cols = checkpoint['targets']
    horizon = checkpoint.get('horizon', 10) # Lấy horizon từ checkpoint, mặc định là 10

    # --- PHẦN SỬA ĐỔI QUAN TRỌNG: TẠO CỘT TARGET ---
    # Model cần so sánh với giá trị ở tương lai (t + horizon)
    for col in target_cols:
        if col not in df.columns:
            # Lấy tên gốc (ví dụ 'temperature' từ 'temperature_t+10')
            base_col = col.split('_t+')[0] 
            if base_col in df.columns:
                # Dịch chuyển dữ liệu ngược lên để lấy giá trị tương lai làm target
                df[col] = df[base_col].shift(-horizon)
    
    # Loại bỏ các dòng cuối bị NaN do không có dữ liệu tương lai để so sánh
    df = df.dropna().reset_index(drop=True)
    # ----------------------------------------------

    if df.empty:
        raise ValueError(f"Dữ liệu sau khi shift bị trống (file quá ngắn so với horizon {horizon})")

    # 3. Chuẩn hóa dữ liệu theo đúng Scaler của model đó
    X = df[features].values
    y = df[target_cols].values
    
    mean = np.array(checkpoint['scaler_mean'])
    scale = np.array(checkpoint['scaler_scale'])
    X_scaled = (X - mean) / scale
    
    X_seq, y_seq = create_sequences(X_scaled, y, checkpoint['seq_len'])
    
    # 4. Dự báo (Inference)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    mean = np.array(checkpoint['scaler_mean'])
    scale = np.array(checkpoint['scaler_scale'])
    preds_original = (preds * scale) + mean
    y_seq_original = (y_seq * scale) + mean
    mae = mean_absolute_error(y_seq_original, preds_original)
    rmse = np.sqrt(mean_squared_error(y_seq_original, preds_original))
    
    # 5. Tính toán sai số
    # mae = mean_absolute_error(y_seq, preds)
    # rmse = np.sqrt(mean_squared_error(y_seq, preds))
    return mae, rmse
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Kiểm tra xem dữ liệu đã được kéo về chưa
    if not os.path.exists(LOCAL_DATA_DIR) or not os.listdir(LOCAL_DATA_DIR):
        print(f"Error: No data found in {LOCAL_DATA_DIR}. Please run mcli first.")
        return

    test_files = sorted([f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith('.csv')])
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])
    
    all_results = []

    print(f"--- Evaluating {len(model_files)} models on {len(test_files)} datasets ---")
    for m_file in model_files:
        m_path = os.path.join(MODEL_DIR, m_file)
        for d_file in test_files:
            d_path = os.path.join(LOCAL_DATA_DIR, d_file)
            try:
                mae, rmse = evaluate_model(m_path, d_path, device)
                all_results.append({
                    "Model": m_file,
                    "Dataset": d_file,
                    "MAE": mae,
                    "RMSE": rmse
                })
                print(f"Done: {m_file} on {d_file}")
            except Exception as e:
                print(f"Skip: Error evaluating {m_file} on {d_file}: {e}")

    # Xuất báo cáo
    report_df = pd.DataFrame(all_results)
    summary = report_df.groupby("Model")[["MAE", "RMSE"]].mean().reset_index()
    summary = summary.sort_values("MAE")
    
    print("\n" + "="*50)
    print("FINAL EVALUATION SUMMARY")
    print("="*50)
    print(summary.to_string(index=False))
    
    report_df.to_csv(RESULTS_PATH, index=False)

if __name__ == "__main__":
    main()