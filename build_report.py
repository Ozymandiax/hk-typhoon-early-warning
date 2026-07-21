import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.tree import DecisionTreeClassifier

HK_LAT = 22.3
HK_LON = 114.2

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組 (陣風突圍版)
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
# 特徵定義: [海平面氣壓(hPa), 24h氣壓降幅(hPa), 10米陣風(km/h)]
# 注意：這裡已經轉用「陣風 (Gusts)」來訓練 AI
X_train = [
    [1010, 0.0, 25], [1008, 1.5, 30], [1006, 2.0, 35],  # 0: 日常陣風 (雷雨/海陸風)
    [1008, -1.0, 20], [1005, 0.5, 35],                  # 0: 氣壓反彈
    [1004, 3.0, 45], [1002, 3.5, 48], [1005, 2.5, 42],  # 1: T1 警戒 (氣壓跌，陣風增強)
    [1000, 5.0, 55], [999, 6.0, 60], [1003, 4.0, 52],   # 3: T3 強風
    # 🌟 陣風 T8 核心邏輯：陣風達 70+，或 (氣壓<1005 且陣風達 60+)
    [1004, 5.0, 65], [1001, 7.0, 75],                   # 8: T8 烈風 (氣壓配合陣風)
    [1008, 3.0, 75], [1006, 4.0, 80]                    # 8: T8 烈風 (純陣風極強)
]
y_train = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8]

# 建立並訓練決策樹
ai_model = DecisionTreeClassifier(max_depth=4, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ AI 決策樹模型訓練完成！")

# ==========================================
# ⚡ 數據獲取與核心計算 (改用陣風 wind_gusts_10m)
# ==========================================
print("⚡ 正在向 Open-Meteo 雲端請求 ECMWF 及 GFS 陣風數據...")
# 🚨 關鍵修改：將 wind_speed_10m 替換為 wind_gusts_10m
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_gusts_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_gusts_10m,pressure_msl&models=gfs_seamless&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf, timeout=15).json()
    res_gfs = requests.get(url_gfs, timeout=15).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

hourly_ec = res_ec['hourly']
times = hourly_ec['time']

# 提取陣風 (Gusts) 數據
ec_wind_keys = [k for k in hourly_ec.keys() if "wind_gusts_10m_member" in k]
ec_press_keys = [k for k in hourly_ec.keys() if "pressure_msl_member" in k]
gfs_wind_keys = [k for k in res_gfs['hourly'].keys() if "wind_gusts_10m_member" in k]
gfs_press_keys = [k for k in res_gfs['hourly'].keys() if "pressure_msl_member" in k]

ec_winds_matrix = np.array([[hourly_ec[k][idx] for k in ec_wind_keys] for idx in range(len(times))]) 
ec_press_matrix = np.array([[hourly_ec[k][idx] for k in ec_press_keys] for idx in range(len(times))]) 
gfs_winds_matrix = np.array([[res_gfs['hourly'][k][idx] for k in gfs_wind_keys] for idx in range(len(times))])
gfs_press_matrix = np.array([[res_gfs['hourly'][k][idx] for k in gfs_press_keys] for idx in range(len(times))])

results = []

for idx, t_str in enumerate(times):
    # 解析 UTC 時間並加上 8 小時轉換為香港時間 (HKT)
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_matrix[idx]  # 這裡現在代表的是「陣風」
    ec_press = ec_press_matrix[idx]
    gfs_winds = gfs_winds_matrix[idx] # 這裡現在代表的是「陣風」
    gfs_press = gfs_press_matrix[idx]
    
    prev_idx = max(0, idx - 24)
    ec_press_drop = ec_press_matrix[prev_idx] - ec_press
    gfs_press_drop = gfs_press_matrix[prev_idx] - gfs_press
    
    # ------------------------------------------
    # 軌道 1：傳統物理門檻判定邏輯 (配合陣風指標)
    # ------------------------------------------
    # T1：氣壓 <= 1006 且 陣風 >= 40 km/h (過濾日常雷雨)
    ec_t1_phy = (ec_press <= 1006) & (ec_winds >= 40)
    gfs_t1_phy = (gfs_press <= 1006) & (gfs_winds >= 40)
    
    # T8：陣風達到烈風下限 63km/h (或氣壓較低時陣風達 55km/h)
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds >= 55)) | (ec_winds >= 65)
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds >= 55)) | (gfs_winds >= 65)
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t1_phy)/len(gfs_press)*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t8_phy)/len(gfs_press)*100)*0.4), 1)

    # ------------------------------------------
    # 軌道 2：🤖 AI 決策樹判定邏輯
    # ------------------------------------------
    ec_ai_preds = []
    for m in range(len(ec_press)):
        feat = [[ec_press[m], ec_press_drop[m], ec_winds[m]]]
        ec_ai_preds.append(ai_model.predict(feat)[0])
        
    gfs_ai_preds = []
    for m in range(len(gfs_press)):
        feat = [[gfs_press[m], gfs_press_drop[m], gfs_winds[m]]]
        gfs_ai_preds.append(ai_model.predict(feat)[0])
        
    ec_ai_preds = np.array(ec_ai_preds)
    gfs_ai_preds = np.array(gfs_ai_preds)
    
    prob_t1_ec_ai = np.sum(ec_ai_preds >= 1) / len(ec_ai_preds) * 100
    prob_t8_ec_ai = np.sum(ec_ai_preds == 8) / len(ec_ai_preds) * 100
    prob_t1_gfs_ai = np.sum(gfs_ai_preds >= 1) / len(gfs_ai_preds) * 100
    prob_t8_gfs_ai = np.sum(gfs_ai_preds == 8) / len(gfs_ai_preds) * 100
    
    # 雙模式 AI 權重集成
    prob_t1_ai = round((prob_t1_ec_ai * 0.6) + (prob_t1_gfs_ai * 0.4), 1)
    prob_t8_ai = round((prob_t8_ec_ai * 0.6) + (prob_t8_gfs_ai * 0.4), 1)

    results.append({
        "時間": display_time,
        "物理一號機率 (%)": prob_t1_phy,
        "物理八號機率 (%)": prob_t8_phy,
        "AI 一號機率 (%)": prob_t1_ai,
        "AI 八號機率 (%)": prob_t8_ai
    })

df_res = pd.DataFrame(results)
df_res_filtered = df_res.iloc[::6, :].reset_index(drop=True)

hkt_now = datetime.now(timezone.utc) + timedelta(hours=8)
update_time_str = hkt_now.strftime('%Y-%m-%d %H:%M')

# ==========================================
# 📊 繪製雙軌對比圖表
# ==========================================
print("⚡ 正在繪製物理與 AI 雙軌集成對比圖...")
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理一號機率 (%)"], name="🟡 傳統物理 (一號風球)", line=dict(color='yellow', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 一號機率 (%)"], name="🌟 AI 決策樹 (一號風球)", line=dict(color='gold', width=3)))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理八號機率 (%)"], name="🔵 傳統物理 (八號風球)", line=dict(color='deepskyblue', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 八號機率 (%)"], name="🔴 AI 決策樹 (八號風球)", line=dict(color='red', width=3)))

fig.update_layout(
    title="🌀 香港風暴信號預測：傳統物理門檻 vs AI 決策樹雙軌制",
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
            <h1>🌀 香港潛在風暴雙軌早期預警系統 (陣風突破版)</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {update_time_str}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>演算法重大升級 (陣風追蹤技術)：</b> 由於全球網格模型會嚴重低估持續風速，本系統已全面切換至 <b>陣風數據 (Wind Gusts)</b> 作為運算核心。當集合模型預測陣風突破 55-65 km/h，並配合氣壓跌幅，系統將精準映射出對應的烈風及八號風球實質威脅機率。
        </div>

        <div>{chart_html}</div>
        
        <div class="table-container">
            <h3>📋 雙軌綜合預測數據表</h3>
            {table_html}
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 恭喜！陣風突圍版 index.html 已成功生成！")
