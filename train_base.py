import itertools
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
import shutil

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# =====================
# TCN MODEL
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
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        y = self.net(x)
        return self.fc(y[:, :, -1])


# =====================
# CONFIG
# =====================
SEQ_LENS = [24]
HORIZONS = [6, 12]
EPOCHS = [30, 50]
BATCH_SIZES = [64, 128]

# ===== TARGETS = FEATURES =====
TARGETS = [
    "temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation",
    "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"
]

DATA_PATH = "dataset_raw/weather_VungTau_2023.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "weather_forecast"


# =====================
# UTILS
# =====================
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)


def prepare_data(df, features, targets, horizon, seq_len):
    df = df.copy()

    # ===== create future targets =====
    for t in targets:
        df[f"{t}_y"] = df[t].shift(-horizon)

    df.dropna(inplace=True)

    # ===== scale input =====
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(df[features])

    # ===== target (NOT scaled – giữ nguyên đơn vị) =====
    y = df[[f"{t}_y" for t in targets]].values

    X_seq, y_seq = create_sequences(X, y, seq_len)
    return X_seq, y_seq, scaler_X


# =====================
# TRAIN ONE MODEL
# =====================
def train_one_model(df, features, cfg):
    name = (
        f"h{cfg['horizon']}_"
        f"ep{cfg['epochs']}_"
        f"bs{cfg['batch_size']}"
    )

    print(f"\n🚀 Training {name}")

    mlflow.start_run(run_name=name)
    try:
        mlflow.log_params(cfg)

        # ===== Prepare data =====
        X, y, scaler_X = prepare_data(
            df, features, TARGETS, cfg["horizon"], cfg["seq_len"]
        )

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True
        )

        # ===== Model =====
        model = TCN(len(features), len(TARGETS))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        loss_values = []

        # ===== Train =====
        for ep in range(cfg["epochs"]):
            model.train()
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

            print(
                f"[{name}] Epoch {ep+1}/{cfg['epochs']} | "
                f"Loss {epoch_loss:.4f}"
            )

        # ===== Log metrics =====
        mlflow.log_metric("mse", epoch_loss)

        # ===== Latency =====
        model.eval()
        with torch.no_grad():
            start = time.time()
            _ = model(X[:1])
            latency = time.time() - start

        mlflow.log_metric("latency_sec", latency)

        # ===== Save model =====
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = f"{MODEL_DIR}/{name}.pth"

        torch.save(
            {
                "state_dict": model.state_dict(),
                "features": features,
                "targets": TARGETS,
                "seq_len": cfg["seq_len"],
                "horizon": cfg["horizon"],
                "scaler_mean": scaler_X.mean_.tolist(),
                "scaler_scale": scaler_X.scale_.tolist(),
                "scaler_X": scaler_X,
                "config": cfg,
            },
            path,
        )

        avg_last = np.mean(loss_values[-5:]) if len(loss_values) >= 5 else np.mean(loss_values)
        mlflow.log_metric("avg_last5_loss", avg_last)

        return avg_last

    finally:
        mlflow.end_run()


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    # ===== FEATURES = TARGETS =====
    FEATURES = TARGETS.copy()

    results = []
    log_lines = []

    for seq_len, horizon, epochs, batch_size in itertools.product(
        SEQ_LENS, HORIZONS, EPOCHS, BATCH_SIZES
    ):
        cfg = {
            "seq_len": seq_len,
            "horizon": horizon,
            "epochs": epochs,
            "batch_size": batch_size,
        }

        loss = train_one_model(df, FEATURES, cfg)

        model_name = f"h{horizon}_ep{epochs}_bs{batch_size}.pth"
        results.append({"model_name": model_name, "loss": loss})

        line = (
            f"seq_len={seq_len}, horizon={horizon}, "
            f"epochs={epochs}, batch_size={batch_size}, "
            f"final_loss={loss:.4f}"
        )
        print(line)
        log_lines.append(line)

    # ===== TOP 3 =====
    top3 = sorted(results, key=lambda x: x["loss"])[:3]

    print("\n TOP 3 MODELS")
    log_lines.append("\nTOP 3 MODELS")
    for i, item in enumerate(top3, 1):
        line = f"{i}. {item['model_name']} | loss={item['loss']:.4f}"
        print(line)
        log_lines.append(line)

    os.makedirs("training_logs", exist_ok=True)
    with open("training_logs/train_results.log", "w", encoding="utf-8") as f:
        for l in log_lines:
            f.write(l + "\n")

    os.makedirs("top3_models", exist_ok=True)
    for item in top3:
        shutil.copy(
            os.path.join(MODEL_DIR, item["model_name"]),
            os.path.join("top3_models", item["model_name"])
        )
