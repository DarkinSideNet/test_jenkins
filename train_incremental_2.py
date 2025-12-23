import itertools
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
import shutil
import glob
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# =====================
# TCN MODEL (Giữ nguyên kiến trúc)
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
# CONFIG (Lấy từ train_base)
# =====================
SEQ_LENS = [24]
HORIZONS = [6, 12]
EPOCHS = [30, 50]
BATCH_SIZES = [8, 16, 32]

TARGETS = [
    "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
    "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
]

# Thư mục chứa model gốc để fine-tune
BASE_MODEL_DIR = "./" 
# Thư mục xuất kết quả incremental
INC_MODEL_DIR = "models_incremental"
EXPERIMENT_NAME = "weather_incremental_training"
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.neikoscloud.net"
# =====================
# UTILS
# =====================
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)

def prepare_data_incremental(df, features, targets, horizon, seq_len, scaler_X):
    df = df.copy()
    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)
    df.dropna(inplace=True)
    
    # Sử dụng scaler cũ để transform data mới (quan trọng trong incremental)
    X = scaler_X.transform(df[features].values)
    y = df[[f"{t}_y" for t in targets]].values
    
    X_seq, y_seq = create_sequences(X, y, seq_len)
    return X_seq, y_seq

# =====================
# TRAIN ONE INCREMENTAL CASE
# =====================
def train_incremental_case(df, features, cfg, base_checkpoint_path):
    name = f"h{cfg['horizon']}_ep{cfg['epochs']}_bs{cfg['batch_size']}"
    print(f"\n🔄 Fine-tuning case: {name}")

    if not os.path.exists(base_checkpoint_path):
        print(f"⚠️ Không tìm thấy model gốc tại {base_checkpoint_path}. Bỏ qua case này.")
        return None
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.start_run(run_name=f"Inc_{name}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_params(cfg)

        # 1. Load Checkpoint gốc
        checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
        
        # Tái tạo Scaler từ checkpoint
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(checkpoint["scaler_mean"])
        scaler_X.scale_ = np.array(checkpoint["scaler_scale"])
        scaler_X.n_features_in_ = len(features)

        # 2. Prepare Data mới
        X_seq, y_seq = prepare_data_incremental(
            df, features, TARGETS, cfg["horizon"], cfg["seq_len"], scaler_X
        )
        
        if len(X_seq) == 0:
            return None

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=cfg["batch_size"], shuffle=True)

        # 3. Load Model
        model = TCN(len(features), len(TARGETS)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # LR thường thấp hơn khi fine-tune
        loss_fn = nn.MSELoss()

        # 4. Fine-tune Loop
        loss_values = []
        model.train()
        for ep in range(cfg["epochs"]):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            loss_values.append(epoch_loss)
            if (ep + 1) % 5 == 0:
                print(f"[{name}] Epoch {ep+1}/{cfg['epochs']} | Loss: {epoch_loss:.4f}")

        # 5. Log & Save
        mlflow.log_metric("mse_inc", epoch_loss)
        
        os.makedirs(INC_MODEL_DIR, exist_ok=True)
        save_path = f"{INC_MODEL_DIR}/{name}.pth"
        
        torch.save({
            "state_dict": model.state_dict(),
            "features": features,
            "targets": TARGETS,
            "seq_len": cfg["seq_len"],
            "horizon": cfg["horizon"],
            "scaler_mean": scaler_X.mean_.tolist(),
            "scaler_scale": scaler_X.scale_.tolist(),
            "config": cfg,
        }, save_path)
        mlflow.log_artifact(save_path, artifact_path="models")

        avg_last = np.mean(loss_values[-5:]) if len(loss_values) >= 5 else np.mean(loss_values)
        return avg_last

    finally:
        mlflow.end_run()

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Lấy file data mới nhất từ dataset_daily
    daily_dir = "./dataset_daily/"
    daily_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    if not daily_files:
        print("❌ Không tìm thấy file csv nào trong dataset_daily.")
        exit(1)
    
    latest_csv = daily_files[-1]
    print(f"📂 Sử dụng dữ liệu mới từ: {latest_csv}")
    df_new = pd.read_csv(latest_csv)

    FEATURES = TARGETS.copy()
    results = []
    log_lines = [f"Incremental Training Log - {time.ctime()}", f"Data source: {latest_csv}\n"]

    # Chạy Grid Search 8 trường hợp
    for seq_len, horizon, epochs, batch_size in itertools.product(
        SEQ_LENS, HORIZONS, EPOCHS, BATCH_SIZES
    ):
        cfg = {
            "seq_len": seq_len,
            "horizon": horizon,
            "epochs": epochs,
            "batch_size": batch_size,
        }

        # Đường dẫn tới model gốc (Giả sử bạn để model gốc trong production_ready 
        # hoặc có thể trỏ thẳng vào thư mục 'models' của train_base)
        # Ở đây tôi mặc định tìm file có cùng 'horizon' vì đó là yếu tố quan trọng nhất của kiến trúc đầu ra
        base_model_path = f"current_model/model.pth" 
        # Nếu bạn muốn mỗi case fine-tune từ đúng case cũ, hãy đổi thành:
        # base_model_path = f"models/h{horizon}_ep{epochs}_bs{batch_size}.pth"

        loss = train_incremental_case(df_new, FEATURES, cfg, base_model_path)

        if loss is not None:
            model_name = f"h{horizon}_ep{epochs}_bs{batch_size}.pth"
            results.append({"model_name": model_name, "loss": loss})
            
            line = f"seq_len={seq_len}, horizon={horizon}, epochs={epochs}, batch_size={batch_size}, final_loss={loss:.4f}"
            print(line)
            log_lines.append(line)

    # ===== XỬ LÝ TOP 3 (Y hệt train_base) =====
    if results:
        top3 = sorted(results, key=lambda x: x["loss"])[:3]

        print("\n🏆 TOP 3 INCREMENTAL MODELS")
        log_lines.append("\nTOP 3 INCREMENTAL MODELS")
        for i, item in enumerate(top3, 1):
            line = f"{i}. {item['model_name']} | loss={item['loss']:.4f}"
            print(line)
            log_lines.append(line)

        # Lưu log
        os.makedirs("training_logs", exist_ok=True)
        with open("training_logs/incremental_results.log", "w", encoding="utf-8") as f:
            for l in log_lines:
                f.write(l + "\n")

        # Copy top 3 vào thư mục riêng
        os.makedirs("top3_models_incremental", exist_ok=True)
        for item in top3:
            shutil.copy(
                os.path.join(INC_MODEL_DIR, item["model_name"]),
                os.path.join("top3_models_incremental", item["model_name"])
            )
        print(f"\n✅ Hoàn thành! Model đã lưu tại {INC_MODEL_DIR} và Top 3 tại top3_models_incremental")