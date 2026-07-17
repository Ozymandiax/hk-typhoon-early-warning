import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

HK_LAT = 22.3
HK_LON = 114.2

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組 (Live Training)
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
# 特徵定義: [海平面氣壓(hPa), 24h氣壓降幅(hPa), 10米風速(km/h)]
# 標籤定義: 0=無信號, 1=一號風球, 3=三號風球, 8=八號風球
X_train = [
    [1012, 0.0, 15], [1010, 0.5, 22], [1008, 1.0, 28],  # 0: 晴朗/日常海陸風
    [1004, 2.5, 25], [1002, 3.0, 32], [1003, 2.8, 20],  # 1: 有颱風胚胎靠近 (T1)
    [1000, 5.0, 42], [999,  6.0, 52], [1001, 4.5, 46],  # 3: 強風圈觸及 (T3)
    [994,  9.0, 65], [988, 14.0, 85], [992, 11.0, 72]   # 8: 核心烈風襲港 (T8)
]
y_train = [0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8]

# 建立並訓練決策樹 (限制深度為 4，防止過度擬合)
ai_model = DecisionTreeClassifier(max_depth=4, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ AI 決策樹模型訓練完成！")

# ==========================================
# ⚡ 數據獲取與核心計算
# ==========================================
print("⚡ 正在向 Open-Meteo 雲端請求 ECMWF 及 GFS 數據...")
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_speed_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={HK_LAT}&longitude={HK_LON}&hourly=wind_speed_10m,pressure_msl&models=gfs_seamless&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf).json()
    res_gfs = requests.get(url_gfs).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

hourly_ec = res_ec['hourly']
times = hourly_ec['time']

ec_wind_keys = [k for k in hourly_ec.keys() if "wind_speed_10m_member" in k]
ec_press_keys = [k for k in hourly_ec.keys() if "pressure_msl_member" in k]
gfs_wind_keys = [k for k in res_gfs['hourly'].keys() if "wind_speed_10m_member" in k]
gfs_press_keys = [k for k in res_gfs['hourly'].keys() if "pressure_msl_member" in k]

ec_winds_matrix = np.array([[hourly_ec[k][idx] for k in ec_wind_keys] for idx in range(len(times))]) 
ec_press_matrix = np.array([[hourly_ec[k][idx] for k in ec_press_keys] for idx in range(len(times))]) 
gfs_winds_matrix = np.array([[res_gfs['hourly'][k][idx] for k in gfs_wind_keys] for idx in range(len(times))])
gfs_press_matrix = np.array([[res_gfs['hourly'][k][idx] for k in gfs_press_keys] for idx in range(len(times))])

results = []
BASE_PRESSURE = 1010.0

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_matrix[idx]
    ec_press = ec_press_matrix[idx]
    gfs_winds = gfs_winds_matrix[idx]
    gfs_press = gfs_press_matrix[idx]
    
    prev_idx = max(0, idx - 24)
    ec_press_drop = ec_press_matrix[prev_idx] - ec_press
    gfs_press_drop = gfs_press_matrix[prev_idx] - gfs_press
    
    # ------------------------------------------
    # 軌道 1：傳統物理門檻判定邏輯 (保留原汁原味)
    # ------------------------------------------
    ec_t1_phy = (ec_press <= 1004) & ((BASE_PRESSURE - ec_press >= 6) | (ec_press_drop >= 2.5))
    ec_t8_phy = (ec_press <= 997) & (ec_winds * 1.2 >= 63)
    gfs_t1_phy = (gfs_press <= 1004) & ((BASE_PRESSURE - gfs_press >= 6) | (gfs_press_drop >= 2.5))
    gfs_t8_phy = (gfs_press <= 997) & (gfs_winds * 1.2 >= 63)
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t1_phy)/len(gfs_press)*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t8_phy)/len(gfs_press)*100)*0.4), 1)

    # ------------------------------------------
    # 軌道 2：🤖 AI 決策樹判定邏輯 (多變數綜合預測)
    # ------------------------------------------
    ec_ai_preds = []
    for m in range(len(ec_press)):
        # 餵入當前成員的特徵: [氣壓, 降幅, 風速(含修正)]
        feat = [[ec_press[m], ec_press_drop[m], ec_winds[m] * 1.1]]
        ec_ai_preds.append(ai_model.predict(feat)[0])
        
    gfs_ai_preds = []
    for m in range(len(gfs_press)):
        feat = [[gfs_press[m], gfs_press_drop[m], gfs_winds[m] * 1.1]]
        gfs_ai_preds.append(ai_model.predict(feat)[0])
        
    ec_ai_preds = np.array(ec_ai_preds)
    gfs_ai_preds = np.array(gfs_ai_preds)
    
    # 計算 AI 預測各級風球的成員比例
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

# ==========================================
# 📊 繪製雙軌對比圖表
# ==========================================
print("⚡ 正在繪製物理與 AI 雙軌集成對比圖...")
fig = go.Figure()

# 畫一號風球對比
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理一號機率 (%)"], name="🟡 傳統物理 (一號風球)", line=dict(color='yellow', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 一號機率 (%)"], name="🌟 AI 決策樹 (一號風球)", line=dict(color='gold', width=3)))

# 畫八號風球對比
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
        .intro-box {{ background: #222; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 14px; line-height: 1.6; border-left: 4px solid #gold; }}
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
            <h1>🌀 香港潛在風暴雙軌早期預警系統 (物理 + AI 決策樹)</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>雙軌運行原理：</b> 本網頁同時展示兩種演算法結果。
            <b>虛線</b>代表「傳統物理硬性門檻」（1004hPa / 997hPa）；
            <b>實線</b>代表「🤖 AI 決策樹機器學習模型」，AI 會綜合評估氣壓值、氣壓驟降斜率與風速的三維交叉關係，減少單一硬性指標嘅死板誤報。
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
print("🎉 恭喜！雙軌 AI 決策樹版 index.html 已成功生成！")
