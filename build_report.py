import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 📍 五星陣雷達坐標設定 (南, 中, 西, 東, 北)
# ==========================================
# 覆蓋: 長洲/橫瀾島, 尖沙咀(核心中樞), 赤鱲角機場, 西貢, 打鼓嶺
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組 (四維特徵終極版)
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
# 特徵定義: [海平面氣壓(hPa), 24h跌幅(hPa), 3h急跌(hPa), 地形修正陣風(km/h)]
X_train = [
    [1010, 0.0,  0.0, 25], [1008, 1.5,  0.5, 30], [1006, 2.0,  0.8, 35],  # 0: 日常陣風
    [1008, -1.0,-0.5, 20], [1005, 0.5,  0.2, 35],                       # 0: 氣壓反彈
    [1004, 3.0,  1.0, 42], [1002, 3.5,  1.5, 45], [1005, 2.5,  0.8, 40],  # 1: T1 警戒
    [1000, 5.0,  2.0, 48], [999,  6.0,  2.5, 50], [1003, 4.0,  1.8, 46],  # 3: T3 強風
    # 🌟 核心突破：加入 3h 急跌特徵，精準捕捉眼牆逼近
    [1004, 5.0,  2.2, 52], [1001, 7.0,  3.0, 55],                       # 8: T8 邊緣直擊 (氣壓急降+陣風 52+)
    [1002, 4.5,  2.5, 50],                                              # 8: 西登擦邊威脅
    [1008, 3.0,  1.5, 62], [1006, 4.0,  2.0, 65]                        # 8: T8 烈風 (純外圍環流強陣風)
]
y_train = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8, 8]

ai_model = DecisionTreeClassifier(max_depth=5, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ 終極四維 AI 決策樹訓練完成！")

# ==========================================
# ⚡ 數據獲取與空間極端值聚合 (加入風向)
# ==========================================
print("⚡ 正在向 Open-Meteo 請求五星陣風、氣壓及核心風向數據...")
# 🚨 API 新增: wind_direction_10m
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,wind_direction_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,wind_direction_10m,pressure_msl&models=gfs_seamless&forecast_days=10"

try:
    res_ec_list = requests.get(url_ecmwf, timeout=15).json()
    res_gfs_list = requests.get(url_gfs, timeout=15).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

times = res_ec_list[0]['hourly']['time']

ec_wind_keys = [k for k in res_ec_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
ec_dir_keys  = [k for k in res_ec_list[0]['hourly'].keys() if "wind_direction_10m_member" in k]
ec_press_keys= [k for k in res_ec_list[0]['hourly'].keys() if "pressure_msl_member" in k]

gfs_wind_keys = [k for k in res_gfs_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
gfs_dir_keys  = [k for k in res_gfs_list[0]['hourly'].keys() if "wind_direction_10m_member" in k]
gfs_press_keys= [k for k in res_gfs_list[0]['hourly'].keys() if "pressure_msl_member" in k]

num_times = len(times)
num_ec_members = len(ec_wind_keys)
num_gfs_members = len(gfs_wind_keys)

# 初始化空間聚合矩陣
ec_winds_max = np.zeros((num_times, num_ec_members))
ec_press_min = np.full((num_times, num_ec_members), 1050.0)
gfs_winds_max = np.zeros((num_times, num_gfs_members))
gfs_press_min = np.full((num_times, num_gfs_members), 1050.0)

# 🌍 五大區域極端值聚合 (Max Gust, Min Pressure)
for loc_data in res_ec_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] for k in ec_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] for k in ec_press_keys] for idx in range(num_times)])
    ec_winds_max = np.maximum(ec_winds_max, loc_winds)
    ec_press_min = np.minimum(ec_press_min, loc_press)

for loc_data in res_gfs_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] for k in gfs_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] for k in gfs_press_keys] for idx in range(num_times)])
    gfs_winds_max = np.maximum(gfs_winds_max, loc_winds)
    gfs_press_min = np.minimum(gfs_press_min, loc_press)

# 🧭 抽取核心中樞 (尖沙咀 Index 1) 的風向，作為地形判定基準
ec_dir_center = np.array([[res_ec_list[1]['hourly'][k][idx] for k in ec_dir_keys] for idx in range(num_times)])
gfs_dir_center = np.array([[res_gfs_list[1]['hourly'][k][idx] for k in gfs_dir_keys] for idx in range(num_times)])

results = []

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_max[idx]
    ec_press = ec_press_min[idx]
    ec_dir   = ec_dir_center[idx]
    
    gfs_winds = gfs_winds_max[idx]
    gfs_press = gfs_press_min[idx]
    gfs_dir   = gfs_dir_center[idx]
    
    # 計算 24 小時與 3 小時氣壓跌幅
    idx_24h = max(0, idx - 24)
    idx_3h  = max(0, idx - 3)
    
    ec_press_drop_24h = ec_press_min[idx_24h] - ec_press
    ec_press_drop_3h  = ec_press_min[idx_3h] - ec_press
    gfs_press_drop_24h = gfs_press_min[idx_24h] - gfs_press
    gfs_press_drop_3h  = gfs_press_min[idx_3h] - gfs_press
    
    # 🏔️ 地形風向懲罰乘數 (Wind Masking)
    # 北/西北偏北 (被大帽山/內陸阻擋): 陣風打 8 折
    # 東至南 (海面完全無遮擋): 陣風乘 1.05
    ec_mult = np.ones_like(ec_winds)
    ec_mult[(ec_dir >= 315) | (ec_dir <= 45)] = 0.8
    ec_mult[(ec_dir >= 90) & (ec_dir <= 180)] = 1.05
    ec_winds_adj = ec_winds * ec_mult
    
    gfs_mult = np.ones_like(gfs_winds)
    gfs_mult[(gfs_dir >= 315) | (gfs_dir <= 45)] = 0.8
    gfs_mult[(gfs_dir >= 90) & (gfs_dir <= 180)] = 1.05
    gfs_winds_adj = gfs_winds * gfs_mult

    # ------------------------------------------
    # 軌道 1：傳統物理門檻判定 (基於地形修正後陣風)
    # ------------------------------------------
    ec_t1_phy = (ec_press <= 1006) & (ec_winds_adj >= 38)
    gfs_t1_phy = (gfs_press <= 1006) & (gfs_winds_adj >= 38)
    
    # 引入 3h 跌幅作為 T8 觸發條件之一 (捕捉急速惡化)
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds_adj >= 48)) | (ec_winds_adj >= 55) | ((ec_press_drop_3h >= 2.0) & (ec_winds_adj >= 45))
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds_adj >= 48)) | (gfs_winds_adj >= 55) | ((gfs_press_drop_3h >= 2.0) & (gfs_winds_adj >= 45))
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t1_phy)/len(gfs_press)*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t8_phy)/len(gfs_press)*100)*0.4), 1)

    # ------------------------------------------
    # 軌道 2：🤖 四維 AI 決策樹判定
    # ------------------------------------------
    ec_ai_preds = []
    for m in range(len(ec_press)):
        feat = [[ec_press[m], ec_press_drop_24h[m], ec_press_drop_3h[m], ec_winds_adj[m]]]
        ec_ai_preds.append(ai_model.predict(feat)[0])
        
    gfs_ai_preds = []
    for m in range(len(gfs_press)):
        feat = [[gfs_press[m], gfs_press_drop_24h[m], gfs_press_drop_3h[m], gfs_winds_adj[m]]]
        gfs_ai_preds.append(ai_model.predict(feat)[0])
        
    ec_ai_preds = np.array(ec_ai_preds)
    gfs_ai_preds = np.array(gfs_ai_preds)
    
    prob_t1_ec_ai = np.sum(ec_ai_preds >= 1) / len(ec_ai_preds) * 100
    prob_t8_ec_ai = np.sum(ec_ai_preds == 8) / len(ec_ai_preds) * 100
    prob_t1_gfs_ai = np.sum(gfs_ai_preds >= 1) / len(gfs_ai_preds) * 100
    prob_t8_gfs_ai = np.sum(gfs_ai_preds == 8) / len(gfs_ai_preds) * 100
    
    prob_t1_ai = round((prob_t1_ec_ai * 0.6) + (prob_t1_gfs_ai * 0.4), 1)
    prob_t8_ai = round((prob_t8_ec_ai * 0.6) + (prob_t8_gfs_ai * 0.4), 1)

    # 計算預測分歧度 (Confidence Spread)
    ec_spread = np.std(ec_winds_adj)
    gfs_spread = np.std(gfs_winds_adj)
    model_spread = round((ec_spread * 0.6) + (gfs_spread * 0.4), 1)

    results.append({
        "時間": display_time,
        "物理一號機率 (%)": prob_t1_phy,
        "物理八號機率 (%)": prob_t8_phy,
        "AI 一號機率 (%)": prob_t1_ai,
        "AI 八號機率 (%)": prob_t8_ai,
        "陣風分歧度 (Uncertainty)": model_spread
    })

df_res = pd.DataFrame(results)
df_res_filtered = df_res.iloc[::6, :].reset_index(drop=True)

hkt_now = datetime.now(timezone.utc) + timedelta(hours=8)
update_time_str = hkt_now.strftime('%Y-%m-%d %H:%M')

# ==========================================
# 📊 繪製終極對比圖表
# ==========================================
print("⚡ 正在繪製終極四維集成對比圖...")
fig = go.Figure()

# Hover Template 增加分歧度顯示
hover_temp = "%{y}%<br>模型分歧度: %{customdata} km/h"

fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理一號機率 (%)"], name="🟡 傳統物理 (一號風球)", line=dict(color='yellow', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 一號機率 (%)"], name="🌟 AI 決策樹 (一號風球)", line=dict(color='gold', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理八號機率 (%)"], name="🔵 傳統物理 (八號風球)", line=dict(color='deepskyblue', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 八號機率 (%)"], name="🔴 AI 決策樹 (八號風球)", line=dict(color='red', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))

fig.update_layout(
    title="🌀 香港風暴信號預測：四維地形修正 AI 雙軌制",
    yaxis_title="發出機率 (%)",
    xaxis_title="預測時間",
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e"
)

table_html = df_res_filtered.to_html(index=False, border=0)
chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港潛在風暴雙軌早期預警系統</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 15px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #333; }}
        h1 {{ margin: 0; color: #ff5252; font-size: 24px; }}
        .update-time {{ color: #888; font-size: 13px; margin-top: 8px; }}
        .intro-box {{ background: #222; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 14px; line-height: 1.6; border-left: 4px solid #ff3333; }}
        .table-container {{ margin-top: 25px; overflow-x: auto; background: #1e1e1e; padding: 15px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: center; font-size: 14px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #333; }}
        th {{ background-color: #2b2b2b; color: #fff; }}
        tr:hover {{ background-color: #252525; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌀 香港潛在風暴雙軌早期預警系統 (終極四維版)</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {update_time_str}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>系統演算法終極升級：</b> 本系統已整合「五星區域極端值聚合」、「風向地形懲罰過濾 (Wind Direction Masking)」及「3小時氣壓急降特徵 ($\Delta P_{3h}$)」。系統不僅能自動捕捉擦邊強風，更懂得根據香港地形智能判斷「有波無風」的假像。將滑鼠懸停於圖表上，更可查看底層集合模型的「陣風分歧度 (Uncertainty)」。
        </div>

        <div>{chart_html}</div>
        
        <div class="table-container">
            <h3>📋 四維綜合預測數據表</h3>
            {table_html}
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 恭喜！終極四維地形修正版 index.html 已成功生成！")
