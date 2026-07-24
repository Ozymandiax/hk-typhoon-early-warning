import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.tree import DecisionTreeClassifier
import json
import os
import math

# ==========================================
# 📍 香港及廣域網格座標設定 (用於精準追蹤與預警)
# ==========================================
HK_CENTER = (22.3193, 114.1694)

# 五星陣雷達坐標 (南, 中, 西, 東, 北)
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
X_train = [
    [1010, 0.0,  0.0, 25], [1008, 1.5,  0.5, 30], [1006, 2.0,  0.8, 35],  # 0: 日常陣風
    [1008, -1.0,-0.5, 20], [1005, 0.5,  0.2, 35],                         # 0: 氣壓反彈
    [1006, 1.0,  1.5, 42],                                                # 0: 新增降噪樣本
    [1004, 3.0,  1.0, 42], [1002, 3.5,  1.5, 45], [1005, 2.5,  0.8, 40],  # 1: T1 警戒
    [1000, 5.0,  2.0, 48], [999,  6.0,  2.5, 50], [1003, 4.0,  1.8, 46],  # 3: T3 強風
    [1004, 5.0,  2.2, 52], [1001, 7.0,  3.0, 55],                         # 8: T8 邊緣直擊
    [1002, 4.5,  2.5, 50],                                                # 8: 西登擦邊威脅
    [1008, 3.0,  1.5, 62], [1006, 4.0,  2.0, 65]                          # 8: T8 烈風
]
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8, 8]

ai_model = DecisionTreeClassifier(max_depth=5, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ 終極四維 AI 決策樹訓練完成！")

# ==========================================
# ⚡ 數據獲取 (Open-Meteo Ensemble API)
# ==========================================
print("⚡ 正在向 Open-Meteo 請求香港周邊 Ensemble 數據...")
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=gfs_seamless&forecast_days=10"
url_dir = "https://api.open-meteo.com/v1/forecast?latitude=22.30&longitude=114.17&hourly=wind_direction_10m&models=ecmwf_ifs025&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf, timeout=15).json()
    res_ec_list = res_ec if isinstance(res_ec, list) else [res_ec]
    
    res_gfs = requests.get(url_gfs, timeout=15).json()
    res_gfs_list = res_gfs if isinstance(res_gfs, list) else [res_gfs]

    res_dir = requests.get(url_dir, timeout=15).json()
    
except Exception as e:
    print(f"❌ API 請求徹底失敗: {e}")
    exit(1)

times = res_ec_list[0]['hourly']['time']
dir_dict = dict(zip(res_dir['hourly']['time'], res_dir['hourly']['wind_direction_10m']))

ec_wind_keys = [k for k in res_ec_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
ec_press_keys= [k for k in res_ec_list[0]['hourly'].keys() if "pressure_msl_member" in k]
gfs_wind_keys = [k for k in res_gfs_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
gfs_press_keys= [k for k in res_gfs_list[0]['hourly'].keys() if "pressure_msl_member" in k]

num_times = len(times)
num_ec_members = len(ec_wind_keys)
num_gfs_members = len(gfs_wind_keys)

ec_winds_max = np.zeros((num_times, num_ec_members))
ec_press_min = np.full((num_times, num_ec_members), 1050.0)
gfs_winds_max = np.zeros((num_times, num_gfs_members))
gfs_press_min = np.full((num_times, num_gfs_members), 1050.0)

for loc_data in res_ec_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] if hourly[k][idx] is not None else 0 for k in ec_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] if hourly[k][idx] is not None else 1050.0 for k in ec_press_keys] for idx in range(num_times)])
    ec_winds_max = np.maximum(ec_winds_max, loc_winds)
    ec_press_min = np.minimum(ec_press_min, loc_press)

for loc_data in res_gfs_list:
    hourly = loc_data['hourly']
    loc_winds = np.array([[hourly[k][idx] if hourly[k][idx] is not None else 0 for k in gfs_wind_keys] for idx in range(num_times)])
    loc_press = np.array([[hourly[k][idx] if hourly[k][idx] is not None else 1050.0 for k in gfs_press_keys] for idx in range(num_times)])
    gfs_winds_max = np.maximum(gfs_winds_max, loc_winds)
    gfs_press_min = np.minimum(gfs_press_min, loc_press)

results = []

# ==========================================
# 核心計算 (風球機率判定)
# ==========================================
for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_max[idx]
    ec_press = ec_press_min[idx]
    gfs_winds = gfs_winds_max[idx]
    gfs_press = gfs_press_min[idx]
    
    center_dir = dir_dict.get(t_str, 90) or 90
        
    idx_24h = max(0, idx - 24)
    idx_3h  = max(0, idx - 3)
    
    ec_press_drop_24h = ec_press_min[idx_24h] - ec_press
    ec_press_drop_3h  = ec_press_min[idx_3h] - ec_press
    gfs_press_drop_24h = gfs_press_min[idx_24h] - gfs_press
    gfs_press_drop_3h  = gfs_press_min[idx_3h] - gfs_press
    
    multiplier = 1.0
    if (center_dir >= 315) or (center_dir <= 45):
        multiplier = 0.8
    elif (center_dir >= 90) and (center_dir <= 180):
        multiplier = 1.05
        
    ec_winds_adj = ec_winds * multiplier
    gfs_winds_adj = gfs_winds * multiplier

    # 軌道 1：傳統物理門檻判定
    ec_t1_phy = (ec_press <= 1005) & (ec_winds_adj >= 38) & (ec_press_drop_24h >= 1.5)
    gfs_t1_phy = (gfs_press <= 1005) & (gfs_winds_adj >= 38) & (gfs_press_drop_24h >= 1.5)
    
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds_adj >= 48)) | (ec_winds_adj >= 55) | ((ec_press_drop_3h >= 2.0) & (ec_winds_adj >= 45))
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds_adj >= 48)) | (gfs_winds_adj >= 55) | ((gfs_press_drop_3h >= 2.0) & (gfs_winds_adj >= 45))
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t1_phy)/len(gfs_press)*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t8_phy)/len(gfs_press)*100)*0.4), 1)

    # 軌道 2：🤖 四維 AI 決策樹判定
    ec_ai_preds = [ai_model.predict([[ec_press[m], ec_press_drop_24h[m], ec_press_drop_3h[m], ec_winds_adj[m]]])[0] for m in range(len(ec_press))]
    gfs_ai_preds = [ai_model.predict([[gfs_press[m], gfs_press_drop_24h[m], gfs_press_drop_3h[m], gfs_winds_adj[m]]])[0] for m in range(len(gfs_press))]
    
    prob_t1_ec_ai = np.sum(np.array(ec_ai_preds) >= 1) / len(ec_ai_preds) * 100
    prob_t8_ec_ai = np.sum(np.array(ec_ai_preds) == 8) / len(ec_ai_preds) * 100
    prob_t1_gfs_ai = np.sum(np.array(gfs_ai_preds) >= 1) / len(gfs_ai_preds) * 100
    prob_t8_gfs_ai = np.sum(np.array(gfs_ai_preds) == 8) / len(gfs_ai_preds) * 100
    
    prob_t1_ai = round((prob_t1_ec_ai * 0.6) + (prob_t1_gfs_ai * 0.4), 1)
    prob_t8_ai = round((prob_t8_ec_ai * 0.6) + (prob_t8_gfs_ai * 0.4), 1)

    ec_spread = np.std(ec_winds_adj) if len(ec_winds_adj) > 0 else 0
    gfs_spread = np.std(gfs_winds_adj) if len(gfs_winds_adj) > 0 else 0
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

# 匯出 Excel 報告
print("📊 正在將預測結果匯出至 Excel...")
os.makedirs("data", exist_ok=True)
df_res_filtered.to_excel("data/typhoon_predictions.xlsx", index=False, engine='openpyxl')

# 結論生成
max_t8_idx = int(df_res_filtered["AI 八號機率 (%)"].idxmax())
max_t8_row = df_res_filtered.iloc[max_t8_idx]

peak_time = max_t8_row["時間"]
peak_prob = max_t8_row["AI 八號機率 (%)"]
peak_spread = max_t8_row["陣風分歧度 (Uncertainty)"]

downgrade_text = ""
if peak_prob >= 20.0:
    after_peak_df = df_res_filtered.iloc[max_t8_idx + 1:]
    downgrade_candidates = after_peak_df[after_peak_df["AI 八號機率 (%)"] < 20.0]
    
    if not downgrade_candidates.empty:
        down_row = downgrade_candidates.iloc[0]
        downgrade_text = f"<br><br>📉 <b>( 8 ➡️ 3) 最有落波時間為香港時間【{down_row['時間']}】，機率為 {down_row['AI 八號機率 (%)']}%</b> <span style='color:#ffaaaa;'>(陣風分歧度：{down_row['陣風分歧度 (Uncertainty)']} km/h)</span>。"
    else:
        downgrade_text = "<br><br>📉 <b>( 8 ➡️ 3) 落波評估：</b>風暴影響時間較長，預測期結束前暫未見明確落波信號。"

if peak_prob >= 20.0:
    conclusion_html = f"""
    <div style="background: linear-gradient(135deg, #4b1313, #8b0000); padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; line-height: 1.6; border-left: 4px solid #ff3333; color: #fff;">
        🚨 <b>AI 實時威脅判定：</b><br>
        根據最新四維運算，預計<b>最有可能懸掛八號風球的時間為【{peak_time}】</b>，
        最高機率達到 <b>{peak_prob}%</b> 
        <span style="color:#ffaaaa;">(陣風分歧度：{peak_spread} km/h)</span>。
        {downgrade_text}
    </div>
    """
elif peak_prob > 0:
    conclusion_html = f"""
    <div style="background: #2b2b00; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; line-height: 1.6; border-left: 4px solid #ffd700; color: #fff;">
        ⚠️ <b>AI 實時威脅判定：</b><br>
        系統目前偵測到八號風球信號，預計高峰期為<b>【{peak_time}】</b>，機率為 <b>{peak_prob}%</b> <span style="color:#aaaaaa;">(陣風分歧度：{peak_spread} km/h)</span>。目前威脅屬於中低度或處於分歧狀態，請密切留意。
    </div>
    """
else:
    conclusion_html = f"""
    <div style="background: #1a2a1a; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; line-height: 1.6; border-left: 4px solid #33ff33; color: #fff;">
        ✅ <b>AI 實時威脅判定：</b><br>
        根據當前數據，未來 10 天內<b>暫未偵測到實質的八號風球威脅</b>。
    </div>
    """

hkt_now = datetime.now(timezone.utc) + timedelta(hours=8)
update_time_str = hkt_now.strftime('%Y-%m-%d %H:%M')

# ==========================================
# 🗺️ 升級：真正的 2D 海洋氣壓網格氣旋追蹤器 (MSLP Tracker)
# ==========================================
print("🗺️ 正在從太平洋/南海網格追蹤氣旋連續軌跡...")

# 提高網格密度 (間隔 1.5 度，覆蓋南海至西太平洋)
grid_lats = [12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5]
grid_lons = [110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0]

grid_lat_str = ",".join(map(str, grid_lats))
grid_lon_str = ",".join(map(str, grid_lons))

url_grid_ec = f"https://api.open-meteo.com/v1/forecast?latitude={grid_lat_str}&longitude={grid_lon_str}&hourly=pressure_msl,wind_speed_10m&models=ecmwf_ifs025&forecast_days=7"

def fetch_and_track_tc():
    features = []
    try:
        res = requests.get(url_grid_ec, timeout=20).json()
        if not isinstance(res, list):
            res = [res]
            
        time_steps = len(res[0]['hourly']['time'])
        tc_track_coords = []
        
        last_lat, last_lon = None, None
        
        for t_idx in range(0, time_steps, 3):  # 每 3 小時採樣一次
            t_str = res[0]['hourly']['time'][t_idx]
            dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
            time_display = dt.strftime("%m-%d %H:00")
            
            min_p = 1018.0
            best_lat, best_lon = None, None
            max_ws = 0
            
            # 搜尋該時間點氣壓最低的網格點
            for pt in res:
                p_val = pt['hourly']['pressure_msl'][t_idx]
                ws_val = pt['hourly']['wind_speed_10m'][t_idx] or 0
                c_lat, c_lon = pt['latitude'], pt['longitude']
                
                # 如果已經有上一個中心，優先搜尋距離上個中心 300km 內的最低氣壓點 (避免跳場)
                if last_lat is not None:
                    dist_from_last = math.hypot(c_lat - last_lat, c_lon - last_lon) * 111
                    if dist_from_last > 350: # 跳太遠就忽略
                        continue
                
                if p_val is not None and p_val < min_p:
                    min_p = p_val
                    best_lat = c_lat
                    best_lon = c_lon
                    max_ws = ws_val
            
            # 只要找到有效低壓中心（<= 1012 hPa），就加入路徑
            if best_lat is not None and min_p <= 1012.0:
                last_lat, last_lon = best_lat, best_lon
                
                # 計算距港距離
                lat1, lon1, lat2, lon2 = map(math.radians, [HK_CENTER[0], HK_CENTER[1], best_lat, best_lon])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                dist_to_hk = round(6371 * 2 * math.asin(math.sqrt(a)))
                
                category, cat_color = classify_tc(max_ws)
                tc_track_coords.append([best_lon, best_lat])
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "time": time_display,
                        "model": "ECMWF 氣旋路徑追蹤",
                        "wind_speed": f"{int(max_ws)} km/h",
                        "pressure": f"{int(min_p)} hPa",
                        "category": category,
                        "color": cat_color,
                        "dist_hk": f"{dist_to_hk} km",
                        "position": f"{best_lat}°N, {best_lon}°E"
                    },
                    "geometry": { "type": "Point", "coordinates": [best_lon, best_lat] }
                })
                
        # 只要採樣點 >= 1 個，就確保畫出連接線與節點
        if len(tc_track_coords) >= 2:
            features.insert(0, {
                "type": "Feature",
                "properties": { "model": "ECMWF 預測路徑", "color": "#00f2fe", "type": "track_line" },
                "geometry": { "type": "LineString", "coordinates": tc_track_coords }
            })
            
    except Exception as e:
        print(f"⚠️ 氣旋網格追蹤分析提示: {e}")
        
    return features

ai_paths_data = {
    "type": "FeatureCollection", 
    "features": fetch_and_track_tc()
}

with open("data/ai_paths.geojson", "w", encoding="utf-8") as f:
    json.dump(ai_paths_data, f, ensure_ascii=False, indent=2)

geojson_json_str = json.dumps(ai_paths_data, ensure_ascii=False)

# ==========================================
# 📊 繪製終極對比圖表
# ==========================================
print("⚡ 正在繪製終極四維集成對比圖...")
fig = go.Figure()
hover_temp = "%{y}%<br>模型分歧度: %{customdata} km/h"

fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理一號機率 (%)"], name="🟡 傳統物理 (一號)", mode='lines+markers', marker=dict(size=6), line=dict(color='yellow', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 一號機率 (%)"], name="🌟 AI 決策樹 (一號)", mode='lines+markers', marker=dict(size=7), line=dict(color='gold', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理八號機率 (%)"], name="🔵 傳統物理 (八號)", mode='lines+markers', marker=dict(size=6), line=dict(color='deepskyblue', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 八號機率 (%)"], name="🔴 AI 決策樹 (八號)", mode='lines+markers', marker=dict(size=7), line=dict(color='red', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))

fig.update_layout(
    title="🌀 香港風暴信號預測：四維地形修正 AI 雙軌制",
    yaxis_title="發出機率 (%)",
    xaxis_title="預測時間",
    yaxis=dict(range=[0, 100]),
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e"
)

table_html = df_res_filtered.to_html(index=False, border=0)
chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# ==========================================
# 🖥️ 修正版 HTML (已徹底 Escape 波浪括號)
# ==========================================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港潛在風暴雙軌早期預警系統</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background-color: #0b132b; color: #e0e0e0; margin: 0; padding: 15px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #1c2541; }}
        h1 {{ margin: 0; color: #ff5252; font-size: 24px; }}
        .update-time {{ color: #888; font-size: 13px; margin-top: 8px; }}
        .intro-box {{ background: #1c2541; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 14px; line-height: 1.6; border-left: 4px solid #4facfe; }}
        .table-container {{ margin-top: 25px; overflow-x: auto; background: #1c2541; padding: 15px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: center; font-size: 14px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #3a506b; }}
        th {{ background-color: #0b132b; color: #fff; }}
        tr:hover {{ background-color: #243256; }}
        
        #typhoon-map {{ width: 100%; height: 550px; border-radius: 10px; background: #0b132b; margin-top: 20px; border: 1px solid #3a506b; }}
        .tc-legend {{ background: rgba(11, 19, 43, 0.9); color: #fff; padding: 12px; border-radius: 8px; font-size: 12px; border: 1px solid #3a506b; line-height: 1.8; }}
        .legend-badge {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌀 香港潛在風暴雙軌早期預警與氣旋追蹤系統</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {update_time_str}</div>
        </div>
        
        {conclusion_html}
        
        <div class="intro-box">
            💡 <b>系統演算法升級：</b> 本系統整合「五星區域極端值聚合」、「風向地形修正」、「3小時/24小時氣壓持續下沉特徵」。下方地圖由 Open-Meteo 廣域海洋 2D 氣壓網格自動追蹤最低氣壓中心，精確呈現熱帶氣旋發育與移動軌迹。
        </div>

        <div>{chart_html}</div>
        
        <div id="typhoon-map"></div>
        
        <div class="table-container">
            <h3>📋 四維綜合預測數據表</h3>
            {table_html}
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('typhoon-map').setView([20.0, 116.5], 6);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 12, minZoom: 4, attribution: '&copy; OpenStreetMap & CartoDB'
        }}).addTo(map);

        // 香港觀察點
        const hkCoords = [22.3193, 114.1694];
        L.circleMarker(hkCoords, {{ radius: 7, color: '#00f2fe', fillColor: '#4facfe', fillOpacity: 1 }}).addTo(map)
         .bindPopup('<b>香港 (Hong Kong)</b><br>預警觀察中心');

        // 香港距離警戒圈 (200km, 400km, 600km)
        const rings = [
            {{ r: 200000, c: '#ef4444', label: '200km 核心警戒圈' }},
            {{ r: 400000, c: '#f97316', label: '400km 近海警戒圈' }},
            {{ r: 600000, c: '#eab308', label: '600km 外圍觀察圈' }}
        ];
        rings.forEach(ring => {{
            L.circle(hkCoords, {{
                radius: ring.r, color: ring.c, weight: 1.2, dashArray: '6, 8', fill: false, opacity: 0.6
            }}).bindTooltip(ring.label).addTo(map);
        }});

        // 載入 GeoJSON 數據
        const geojsonData = {geojson_json_str};

        if (geojsonData.features.length > 0) {{
            L.geoJSON(geojsonData, {{
                style: function(feature) {{
                    if (feature.properties.type === "track_line") {{
                        return {{ color: feature.properties.color, weight: 3, opacity: 0.85, dashArray: '4, 4' }};
                    }}
                }},
                pointToLayer: function(feature, latlng) {{
                    return L.circleMarker(latlng, {{
                        radius: 6,
                        fillColor: feature.properties.color || '#ffffff',
                        color: '#000000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.95
                    }});
                }},
                onEachFeature: function(feature, layer) {{
                    if (feature.geometry.type === "Point") {{
                        const p = feature.properties;
                        layer.bindPopup(
                            '<div style="font-size:13px; line-height:1.5;">' +
                            '<b style="color:' + p.color + ';">' + p.category + ' (' + p.wind_speed + ')</b><br>' +
                            '<b>時間：</b>' + p.time + '<br>' +
                            '<b>中心氣壓：</b>' + p.pressure + '<br>' +
                            '<b>距港距離：</b>' + p.dist_hk + '<br>' +
                            '<b>座標位置：</b>' + p.position +
                            '</div>'
                        );
                    }}
                }}
            }}).addTo(map);
        }}

        // 圖例
        const legend = L.control({{ position: 'topright' }});
        legend.onAdd = function () {{
            const div = L.DomUtil.create('div', 'tc-legend');
            div.innerHTML = `
                <b>🌀 熱帶氣旋等級 (標準)</b><br>
                <span class="legend-badge" style="background:#3b82f6;"></span>熱帶低氣壓 (&lt;41 km/h)<br>
                <span class="legend-badge" style="background:#06b6d4;"></span>熱帶風暴 (41-62 km/h)<br>
                <span class="legend-badge" style="background:#eab308;"></span>強烈熱帶風暴 (63-87 km/h)<br>
                <span class="legend-badge" style="background:#f97316;"></span>颱風 (88-117 km/h)<br>
                <span class="legend-badge" style="background:#ef4444;"></span>強颱風 (118-149 km/h)<br>
                <span class="legend-badge" style="background:#a855f7;"></span>超強颱風 (&ge;150 km/h)
            `;
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 恭喜！網格氣旋追蹤與修正版 Dashboard HTML 已完美生成！")
