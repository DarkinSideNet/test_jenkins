import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




# =========================
# CẤU HÌNH RISK (tuỳ chỉnh)
# =========================

# Nếu có dữ liệu camera (tỉ lệ vùng nước nhìn thấy 0..1), có thể truyền qua ENV hoặc set cứng tại đây:
#CAM_WATER_PCT = float(os.getenv("CAM_WATER_PCT", "0.0"))  # ví dụ 0.3 = 30%

#tham chiếu cho risk theo mưa/humidity (khi không có water level)
R_REF_MM = 30.0      # mưa tham chiếu trong 1 bước (mm)
HUM_FLOOR = 70.0     # %RH từ ngưỡng này trở lên coi như đất dễ bão hoà (tuỳ địa phương)
HUM_RANGE = 30.0     # 70% -> 100% sẽ map về 0..1 (70 + 30 = 100)

#trọng số khi không có water level
W_RAIN = 0.6
W_HUM = 0.20
W_WIND = 0.20

# Tham số khi CÓ water_level (nếu sau này model dự báo thêm)
H_WARN = 0.6   # m, mốc cảnh báo
H_CRIT = 0.8   # m, mốc ngập
W_BASE = 0.55
W_TREND = 0.15
W_RAIN_LVL = 0.20
W_CAM_LVL  = 0.10
DHref = 0.05  # m/step tham chiếu cho trend



# ===============
# HÀM TIỆN ÍCH
# ===============
def clamp01(x:float)->float:
    return max(0.0, min(1.0, float(x)))
# ---------------------------
# RISK từ mưa + độ ẩm + camera
# (dùng khi KHÔNG có water_level)
# ---------------------------

def risk_from_rain_humidi(pred_map: dict, cam_water_pct: float = 0.0,r_ref_mm: float = 30.0,
                          hum_floor: float = 70.0,
                          hum_range: float = 30.0,
                          w_rain: float = 0.6,
                           w_hum: float = 0.2,
                           w_wind: float = 0.2,)->dict:
    """
    pred_map: {'rain': value_mm, 'humidi': value_percent, ...} dự báo cho bước tới
    Trả về risk score 0..100 dựa trên mưa + độ ẩm (proxy bão hoà) + camera.
    """

    rain = float(pred_map.get("rain", 40.0))
    hum = float(pred_map.get("humidi", 50.0)) # giả định 50% nếu thiếu
    wind = float(pred_map.get("wind", 10.0))


    rain_term = clamp01(rain / max(1e-6, r_ref_mm))
    hum_term  = clamp01((hum - hum_floor) / max(1e-6, hum_range))
    wind_term = clamp01(wind /50.0)  # giả sử 50 m/s là gió rất mạnh



    s = w_rain + w_hum + w_wind
    w_rain_eff = w_rain / s
    w_hum_eff  = w_hum  / s
    w_wind_eff = w_wind / s
    score_lin = w_rain_eff*rain_term + w_hum_eff*hum_term + w_wind_eff*wind_term

    # if cam_term == 0.0 or w_cam == 0.0:
    #     s = w_rain + w_hum
    #     if s > 0:
    #         w_rain_eff = w_rain / s
    #         w_hum_eff  = w_hum  / s
    #     else:
    #         w_rain_eff, w_hum_eff = 0.5, 0.5
    #     score_lin = w_rain_eff * rain_term + w_hum_eff * hum_term
    # else:
    #     score_lin = w_rain * rain_term + w_hum * hum_term + w_cam * cam_term

    score = int(round(100 * clamp01(score_lin)))

    level = "Low" if score < 30 else "Medium" if score < 60 else "High" if score < 80 else "Severse"
    return {
        "risk_score": score,
        "risk_level": level,
        "rain_term": float(rain_term),
        "hum_term": float(hum_term),
        "wind_term": float(wind_term),
        "explain": "Risk dựa trên mưa dự báo, độ ẩm (proxy bão hoà)"
    }
# ----------------------------------------------------
# RISK từ chuỗi mực nước (nếu SAU NÀY bạn có water_level)
# (giữ nguyên ở đây để bạn dễ nâng cấp khi có thêm target)
# ----------------------------------------------------
def risk_from_water_level(
    H_pred_paths,           # np.array shape (N,T) nếu MC Dropout; hoặc (T,) nếu 1 đường dự báo
    H_warn: float, H_crit: float,
    R_forecast_mm: float = 0.0,
    cam_water_pct: float = 0.0,
    w_base: float = 0.55, w_trend: float = 0.15,
    w_rain: float = 0.20, w_cam: float = 0.10,
    R_ref: float = 30.0,
    dHref: float = 0.05,
    hysteresis_prev_score: int | None = None,
    hysteresis_decay: float = 0.9
):
    H_pred_paths = np.asarray(H_pred_paths)
    if H_pred_paths.ndim == 1:
        H_pred_paths = H_pred_paths[None, :]  # (1,T)

    max_levels = H_pred_paths.max(axis=1)
    p = float((max_levels >= H_crit).mean())

    H_med = np.median(H_pred_paths, axis=0)
    base = clamp01(np.max((H_med - H_warn) / max(1e-6, (H_crit - H_warn))))

    trend = 0.0
    dH = np.diff(H_med)
    if len(dH) > 0:
        trend = clamp01(np.max(dH) / max(1e-6, dHref))

    rain_term = clamp01(R_forecast_mm / max(1e-6, R_ref))
    cam_term  = clamp01(cam_water_pct)

    score_linear = clamp01(p + 0.5 * (w_base*base + w_trend*trend + w_rain*rain_term + w_cam*cam_term))
    score = int(round(100 * score_linear))
    if hysteresis_prev_score is not None and score < hysteresis_prev_score:
        score = int(round(hysteresis_decay * hysteresis_prev_score + (1-hysteresis_decay) * score))

    level = "Low" if score < 30 else "Medium" if score < 60 else "High" if score < 80 else "Severe"
    eta_step = int(np.argmax(H_med >= H_crit)) if (H_med.max() >= H_crit) else None

    return {
        "risk_score": score,
        "risk_level": level,
        "prob_exceed": p,
        "eta_step": eta_step,
        "base": float(base),
        "trend": float(trend),
        "rain_term": float(rain_term),
        "cam_term": float(cam_term),
        "explain": "Risk dựa trên phân phối mực nước + điều chỉnh mưa & camera."
    }


def load_ckpt(pt_path="tcn_best_multi.pt"):
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
    # Import lớp TCN đúng file train của bạn
    try:
        from weather_train_multi import TCN  # nếu bạn đặt tên file train là weather_train_multi.py
    except ImportError:
        from weather_train import TCN        # fallback nếu file là weather_train.py
    
    # Dựng model với đúng số kênh input và số output
    model = TCN(
        in_ch=len(ckpt["features"]),
        chans=ckpt["tcn_channels"],
        ks=ckpt["tcn_kernel_size"],
        dropout=ckpt["tcn_dropout"],
        n_targets=len(targets),
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
    x = torch.from_numpy(last_seq_scaled).unsqueeze(0).transpose(1, 2)

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
    model, scaler, ckpt, targets = load_ckpt("tcn_best_multi.pt")
     # Nạp dữ liệu mới từ CSV hoặc DataFrame build từ MQTT/DB
    df_new = pd.read_csv(r"weather_test_2.csv")  # đường dẫn tới CSV mới thu thập
    res = predict_with_new_data(df_new, model, scaler, ckpt, targets)
    # In ra 2 giá trị: rain, humidi (nếu tồn tại trong targets)
    preds = res["prediction"]
    # Ưu tiên in theo thứ tự mong muốn
    for name in ["temperature", "feels_like", "humidity", "wind_speed", "gust_speed", "pressure", "precipitation", "rain_probability", "snow_probability", "uv_index", "dewpoint", "visibility", "cloud"]:
        if name in preds:
            print(f"{name}: {preds[name]:.3f}")



    # # ==== tính risk score =====
    # # ứu tiên water_level nếu có trong target
    # if "water_level" in preds:
    #     H_pred = np.array([preds["water_level"]], dtype=float)
    #     risk = risk_from_water_level(
    #         H_pred_paths=H_pred,
    #         H_warn=H_WARN,
    #         H_crit=H_CRIT,
    #         R_forecast_mm=float(preds.get("rain", 0.0)),

    #         w_base=W_BASE, w_trend=W_TREND,
    #         w_rain=W_RAIN_LVL, w_cam=W_CAM_LVL,
    #         R_ref=R_REF_MM, dHref=DHref
    #     )
    # else:
    #     risk = risk_from_rain_humidi(
    #         pred_map=preds,
 
    #         r_ref_mm=R_REF_MM,
    #         hum_floor=HUM_FLOOR,
    #         hum_range=HUM_RANGE,
    #         w_rain=W_RAIN,
    #         w_hum=W_HUM,
    #         w_wind=W_WIND
    #     )
    # print ("----Risk Score----")
    # print(f"risk_score: {risk['risk_score']}  ({risk['risk_level']})")
    # if "prob_exceed" in risk:
    #     print(f"prob_exceed: {risk['prob_exceed']:.2f}")
    # if "rain_term" in risk: print(f"rain_term: {risk['rain_term']:.2f}")
    # if "hum_term"  in risk: print(f"hum_term : {risk.get('hum_term', 0.0):.2f}")
    # if "wind_term"  in risk: print(f"wind_term : {risk['wind_term']:.2f}")
    # if "eta_step"  in risk and risk["eta_step"] is not None:
    #     print(f"ETA vượt H_CRIT ở bước: {risk['eta_step']}")
    # print(risk["explain"])
    # # Nếu muốn thấy tất cả target theo thứ tự ckpt:
    # # print("---- Tất cả targets ----")
    # # for t in res["targets"]:
    #     # print(f"{t}: {preds[t]:.3f}")
    # # print(f"(horizon={res['horizon']}, seq_len={res['seq_len']})")

if __name__ == "__main__":
    main()
