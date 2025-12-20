import subprocess
import os
import json
import shutil
import sys
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Tạo thư mục logs nếu chưa có
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Đặt tên file log theo ngày giờ chạy
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"train_session_{current_time}.log")

# Chuyển hướng đầu ra hệ thống vào file log
sys.stdout = Logger(log_file)
# -------------------------------------

print(f"--- Session started at: {current_time} ---")
print(f"--- All outputs will be saved to: {log_file} ---")
def run_multi_experiments():
    experiments = [
        {"epochs": 10, "bs": 64,  "lr": 1e-3, "horizon": 10},
        {"epochs": 20, "bs": 64,  "lr": 5e-4, "horizon": 10},
        {"epochs": 15, "bs": 128, "lr": 1e-3, "horizon": 10},
        {"epochs": 30, "bs": 128, "lr": 5e-4, "horizon": 10},
        {"epochs": 10, "bs": 32,  "lr": 2e-4, "horizon": 10},
        {"epochs": 25, "bs": 256, "lr": 1e-3, "horizon": 10},
        {"epochs": 20, "bs": 128, "lr": 8e-4, "horizon": 10},
        {"epochs": 40, "bs": 64,  "lr": 1e-4, "horizon": 10},
    ]
    results = []
    os.makedirs("./exp_results", exist_ok=True)

    for i, exp in enumerate(experiments):
        out_path = f"./exp_results/model_exp_{i}.pth"
        data_path = "./dataset.csv"
        print(f"\n>>> Starting Experiment {i}: {exp}")
        cmd = [
            "python", "weather_train_fire_tuning.py",
            "--data_path", data_path,
            "--output_path", out_path,
            "--epochs", str(exp["epochs"]),
            "--batch_size", str(exp["bs"]),
            "--lr", str(exp["lr"]),
            "--horizon", str(exp["horizon"]),
        ]
        # Chạy và chờ kết quả
        subprocess.run(cmd, check=True)

        import torch
        checkpoint = torch.load(out_path)
        val_loss = checkpoint.get("val_loss", float('inf'))
        
        results.append({
            "id": i,
            "params": exp,
            "val_loss": val_loss,
            "path": out_path
        })
    results.sort(key=lambda x: x["val_loss"])
    top_3 = results[:3]

    print("\n" + "="*30)
    print("TOP 3 BEST MODELS FOUND:")
    os.makedirs("./best_models", exist_ok=True)

    for rank, item in enumerate(top_3):
        print(f"Rank {rank+1}: Exp {item['id']} | Loss: {item['val_loss']:.6f}")
        # Copy vào thư mục lưu trữ cuối cùng
        shutil.copy(item["path"], f"./best_models/top_{rank+1}_model.pth")

if __name__ == "__main__":
    run_multi_experiments()
