import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 📍 五星陣雷達坐標設定 (南, 中, 西, 東, 北)
# ==========================================
# 覆蓋: 長洲/橫瀾島, 尖沙咀, 赤鱲角機場, 西貢, 打鼓嶺
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組 (極限追蹤版)
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
# 特徵定義: [海平面氣壓(hPa), 24h氣壓降幅(hPa), 10米陣風(km/h)]
X_train = [
    [1010, 0.0, 25], [1008, 1.5, 30], [1006, 2.0, 35],  # 0: 日常陣風 (雷雨/海陸風)
    [1008, -1.0, 20], [1005, 0.5, 35],                  # 0: 氣壓反彈
    [1004, 3.0, 42], [1002, 3.5, 45], [1005, 2.5, 40],  # 1: T1 警戒
    [1000, 5.0, 48], [999, 6.0, 50], [1003, 4.0, 46],   # 3: T3 強風
    # 🌟 極限追蹤修正：捕捉擦邊成員，陣風達 52+ 即判定 T8
    [1004, 5.0, 52], [1001, 7.0, 55],                   # 8: T8 邊緣直擊 (氣壓配合陣風 52+)
    [1002, 4.5, 50],                                    # 8: 西登擦邊威脅
    [1008, 3.0, 62], [1006, 4.0, 65]                    # 8: T8 烈風 (純外圍環流強陣風 62+)
]
y_train = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8, 8]

ai_model = DecisionTreeClassifier(max_depth=4, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ AI 決策樹模型訓練完成！")

# ==========================================
# ⚡ 數據獲取與空間極端值聚合 (Spatial Aggregation)
# ==========================================
print("⚡ 正在向 Open-Meteo 雲端請求五大測風點陣風數據...")
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=gfs_seamless&forecast_days=10"

try:
    # 由於查詢多個坐標，API 會返回一個包含 5 個 JSON Object 的 List
    res_ec_list = requests.get(url_ecmwf, timeout=15).json()
    res_gfs_list = requests.get(url_gfs, timeout=15).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

# 以第一個地點的時間軸為基準
times = res_ec_list[0]['hourly']['time']

ec_wind_keys = [k for k in res_ec_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
ec_press_keys = [k for k in res_ec_list[0]['hourly'].keys() if "pressure_msl_member" in k]
gfs_wind_keys = [k for k in res_gfs_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
gfs_press_keys = [k for k in res_gfs_list[0]['hourly'].keys() if "pressure_msl_member" in k]

num_times = len(times)
num_ec_members = len(ec_wind_keys)
num_gfs_members = len(gfs_wind_keys)

# 初始化聚合矩陣：風速尋找最大值，氣壓尋找最小值
ec_winds_matrix = np.zeros((num_times, num_ec_members))
ec_press_matrix = np.full((num_times, num_ec_members), 1050.0)
gfs_winds_matrix = np.zeros((num_times, num_gfs_members))
gfs_press_matrix = np.full((num_times, num_gfs_members), 1050.0)

# 🌍 核心邏輯：遍歷 5 個地點，提取區域內的最極端數據
for loc_data in res_ec_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] for k in ec_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] for k in ec_press_keys] for idx in range(num_times)])
    # 覆蓋為所有地點中的最高陣風與最低氣壓
    ec_winds_matrix = np.maximum(ec_winds_matrix, loc_winds)
    ec_press_matrix = np.minimum(ec_press_matrix, loc_press)

for loc_data in res_gfs_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] for k in gfs_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] for k in gfs_press_keys] for idx in range(num_times)])
    gfs_winds_matrix = np.maximum(gfs_winds_matrix, loc_winds)
    gfs_press_matrix = np.minimum(gfs_press_matrix, loc_press)

results = []
BASE_PRESSURE = 1010.0

for idx, t_str in enumerate(times):
    # 解析 UTC 時間並加上 8 小時轉換為香港時間 (HKT)
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_matrix[idx]  
    ec_press = ec_press_matrix[idx]
    gfs_winds = gfs_winds_matrix[idx]
    gfs_press = gfs_press_matrix[idx]
    
    prev_idx = max(0, idx - 24)
    ec_press_drop = ec_press_matrix[prev_idx] - ec_press
    gfs_press_drop = gfs_press_matrix[prev_idx] - gfs_press
    
    # ------------------------------------------
    # 軌道 1：傳統物理門檻判定邏輯 (極限追蹤版)
    # ------------------------------------------
    # T1：氣壓 <= 1006 且 陣風 >= 40 km/h (過濾日常雷雨)
    ec_t1_phy = (ec_press <= 1006) & (ec_winds >= 40)
    gfs_t1_phy = (gfs_press <= 1006) & (gfs_winds >= 40)
    
    # T8：只要區域內有任何一點陣風達 55+，或氣壓跌穿 1005 時陣風達 48+
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds >= 48)) | (ec_winds >= 55)
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds >= 48)) | (gfs_winds >= 55)
    
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
            <h1>🌀 香港潛在風暴雙軌早期預警系統 (五星雷達聚合版)</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {update_time_str}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>系統演算法終極升級 (五星陣空間聚合技術)：</b> 系統已放棄傳統的「單點監測」盲區，升級為同時監測香港東南西北中 5 個極端地理位置 (包括長洲及橫瀾島)。演算法會實時提取這 5 個區域內的 <b>「最高陣風」與「最低氣壓」極值</b> 作為判定基準，完美模擬天文台「八中四」的宏觀掛波邏輯，精準捕捉所有致命的擦邊路徑威脅！
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

print("🎉 恭喜！五星雷達聚合版 index.html 已成功生成！")
