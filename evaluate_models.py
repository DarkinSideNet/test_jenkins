import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import argparse

LOCAL_DATA_DIR = "./evaluation_data"
MODEL_DIR = "./best_models"
RESULTS_PATH = "./evaluation_results.csv"

# [Copy các class WeatherDataset, TCNBlock, TCN từ train.py vào đây để script có thể rebuild model]

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)
def evaluate_model(model_path, data_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Rebuild model dựa trên kiến trúc đã lưu
    model = TCN(num_inputs=len(checkpoint['features']), 
                num_outputs=len(checkpoint['targets']), 
                channels=[32, 64, 128], kernel_size=3, dropout=0.2).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    df = pd.read_csv(data_path)
    features = checkpoint['features']
    target_cols = checkpoint['targets']
    
    # Chuẩn hóa dữ liệu theo đúng Scaler của model đó
    X = df[features].values
    y = df[target_cols].values
    
    mean = np.array(checkpoint['scaler_mean'])
    scale = np.array(checkpoint['scaler_scale'])
    X_scaled = (X - mean) / scale
    
    X_seq, y_seq = create_sequences(X_scaled, y, checkpoint['seq_len'])
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    
    mae = mean_absolute_error(y_seq, preds)
    rmse = np.sqrt(mean_squared_error(y_seq, preds))
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