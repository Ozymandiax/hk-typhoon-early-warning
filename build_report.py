import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

HK_LAT = 22.3
HK_LON = 114.2

print("⚡ 正在向 Open-Meteo 雲端請求 ECMWF 及 GFS 氣壓與風速數據...")

# 同時請求 wind_speed_10m (風速) 和 pressure_msl (海平面氣壓)
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_speed_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_speed_10m,pressure_msl&models=gfs_seamless&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf).json()
    res_gfs = requests.get(url_gfs).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

print("⚡ 數據獲取成功！正在進行物理邏輯修正與多模式集成計算...")

hourly_ec = res_ec['hourly']
times = hourly_ec['time']

# 獲取成員 key 列表
ec_wind_keys = [k for k in hourly_ec.keys() if "wind_speed_10m_member" in k]
ec_press_keys = [k for k in hourly_ec.keys() if "pressure_msl_member" in k]

gfs_wind_keys = [k for k in res_gfs['hourly'].keys() if "wind_speed_10m_member" in k]
gfs_press_keys = [k for k in res_gfs['hourly'].keys() if "pressure_msl_member" in k]

results = []

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
    display_time = dt.strftime("%m月%d日 %H:00")
    
    # --- 1. ECMWF 成員數據提取 ---
    ec_winds = np.array([hourly_ec[k][idx] for k in ec_wind_keys if hourly_ec[k][idx] is not None])
    ec_press = np.array([hourly_ec[k][idx] for k in ec_press_keys if hourly_ec[k][idx] is not None])
    
    # --- 2. GFS 成員數據提取 ---
    gfs_winds = np.array([res_gfs['hourly'][k][idx] for k in gfs_wind_keys if res_gfs['hourly'][k][idx] is not None])
    gfs_press = np.array([res_gfs['hourly'][k][idx] for k in gfs_press_keys if res_gfs['hourly'][k][idx] is not None])
    
    # --- 3. 嚴格的風球判定邏輯 (必須結合氣壓下降，避免日常海陸風干擾) ---
    # T1：氣壓低於 1006 hPa (代表有氣旋/低壓槽逼近)
    prob_t1_ec = float(np.sum(ec_press <= 1006) / len(ec_press) * 100) if len(ec_press) > 0 else 0
    # T3：氣壓低於 1004 hPa 且風速(含1.1倍地形修正) >= 41 km/h
    prob_t3_ec = float(np.sum((ec_press <= 1004) & (ec_winds * 1.1 >= 41)) / len(ec_winds) * 100) if len(ec_winds) > 0 else 0
    # T8：氣壓低於 1000 hPa 且風速(含1.2倍地形修正) >= 63 km/h
    prob_t8_ec = float(np.sum((ec_press <= 1000) & (ec_winds * 1.2 >= 63)) / len(ec_winds) * 100) if len(ec_winds) > 0 else 0

    # GFS 同理
    prob_t1_gfs = float(np.sum(gfs_press <= 1006) / len(gfs_press) * 100) if len(gfs_press) > 0 else 0
    prob_t3_gfs = float(np.sum((gfs_press <= 1004) & (gfs_winds * 1.1 >= 41)) / len(gfs_winds) * 100) if len(gfs_winds) > 0 else 0
    prob_t8_gfs = float(np.sum((gfs_press <= 1000) & (gfs_winds * 1.2 >= 63)) / len(gfs_winds) * 100) if len(gfs_winds) > 0 else 0

    # --- 4. 權重集成 (60% EC + 40% GFS) ---
    prob_t1_ensemble = round((prob_t1_ec * 0.6) + (prob_t1_gfs * 0.4), 1)
    prob_t3_ensemble = round((prob_t3_ec * 0.6) + (prob_t3_gfs * 0.4), 1)
    prob_t8_ensemble = round((prob_t8_ec * 0.6) + (prob_t8_gfs * 0.4), 1)

    results.append({
        "時間": display_time,
        "ECMWF 8號機率 (%)": round(prob_t8_ec, 1),
        "GFS 8號機率 (%)": round(prob_t8_gfs, 1),
        "綜合集成 1號機率 (%)": prob_t1_ensemble,
        "綜合集成 3號機率 (%)": prob_t3_ensemble,
        "綜合集成 8號機率 (%)": prob_t8_ensemble
    })

df_res = pd.DataFrame(results)

# 為了網頁乾淨，每 6 小時顯示一檔
df_res_filtered = df_res.iloc[::6, :].reset_index(drop=True)

# 繪圖邏輯保持不變... (此處省略後續繪圖代碼，請直接套用原先 build_report.py 的繪圖與輸出部分)
