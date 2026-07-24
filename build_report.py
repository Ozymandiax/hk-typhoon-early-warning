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
# 📍 五星陣雷達坐標設定 (南, 中, 西, 東, 北)
# ==========================================
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"
HK_CENTER = (22.3, 114.2) # 香港中心坐標，用於 AI 路徑推算

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
X_train = [
    [1010, 0.0,  0.0, 25], [1008, 1.5,  0.5, 30], [1006, 2.0,  0.8, 35], 
    [1008, -1.0,-0.5, 20], [1005, 0.5,  0.2, 35], [1006, 1.0,  1.5, 42], 
    [1004, 3.0,  1.0, 42], [1002, 3.5,  1.5, 45], [1005, 2.5,  0.8, 40], 
    [1000, 5.0,  2.0, 48], [999,  6.0,  2.5, 50], [1003, 4.0,  1.8, 46], 
    [1004, 5.0,  2.2, 52], [1001, 7.0,  3.0, 55], [1002, 4.5,  2.5, 50], 
    [1008, 3.0,  1.5, 62], [1006, 4.0,  2.0, 65] 
]
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8, 8]

ai_model = DecisionTreeClassifier(max_depth=5, random_state=42)
ai_model.fit(X_train, y_train)

# ==========================================
# ⚡ 數據獲取 (Open-Meteo API)
# ==========================================
print("⚡ 正在向 Open-Meteo 請求數據...")
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=ecmwf_ifs04&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=gfs_seamless&forecast_days=10"
url_dir = "https://api.open-meteo.com/v1/forecast?latitude=22.30&longitude=114.17&hourly=wind_direction_10m&models=ecmwf_ifs04&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf, timeout=15).json()
    res_gfs = requests.get(url_gfs, timeout=15).json()
    res_dir = requests.get(url_dir, timeout=15).json()
except Exception as e:
    print(f"❌ API 請求失敗: {e}")
    exit(1)

res_ec_list = res_ec if isinstance(res_ec, list) else [res_ec]
res_gfs_list = res_gfs if isinstance(res_gfs, list) else [res_gfs]

times = res_ec_list[0]['hourly']['time']
dir_dict = dict(zip(res_dir['hourly']['time'], res_dir['hourly']['wind_direction_10m']))

ec_wind_keys = [k for k in res_ec_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
ec_press_keys= [k for k in res_ec_list[0]['hourly'].keys() if "pressure_msl_member" in k]
gfs_wind_keys = [k for k in res_gfs_list[0]['hourly'].keys() if "wind_gusts_10m_member" in k]
gfs_press_keys= [k for k in res_gfs_list[0]['hourly'].keys() if "pressure_msl_member" in k]

num_times = len(times)
ec_winds_max = np.zeros((num_times, len(ec_wind_keys)))
ec_press_min = np.full((num_times, len(ec_press_keys)), 1050.0)
gfs_winds_max = np.zeros((num_times, len(gfs_wind_keys)))
gfs_press_min = np.full((num_times, len(gfs_press_keys)), 1050.0)

for loc_data in res_ec_list:
    hourly = loc_data['hourly']
    ec_winds_max = np.maximum(ec_winds_max, np.array([[hourly[k][idx] or 0 for k in ec_wind_keys] for idx in range(num_times)]))
    ec_press_min = np.minimum(ec_press_min, np.array([[hourly[k][idx] or 1050.0 for k in ec_press_keys] for idx in range(num_times)]))

for loc_data in res_gfs_list:
    hourly = loc_data['hourly']
    gfs_winds_max = np.maximum(gfs_winds_max, np.array([[hourly[k][idx] or 0 for k in gfs_wind_keys] for idx in range(num_times)]))
    gfs_press_min = np.minimum(gfs_press_min, np.array([[hourly[k][idx] or 1050.0 for k in gfs_press_keys] for idx in range(num_times)]))

results = []

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_max[idx]
    ec_press = ec_press_min[idx]
    gfs_winds = gfs_winds_max[idx]
    gfs_press = gfs_press_min[idx]
    
    center_dir = dir_dict.get(t_str, 90) or 90
    
    # 🔥 關鍵修復：嚴格限制時間窗，防止「大氣潮汐」假陽性誤判
    ec_press_drop_24h = ec_press_min[idx - 24] - ec_press if idx >= 24 else 0.0
    ec_press_drop_3h  = ec_press_min[idx - 3] - ec_press if idx >= 3 else 0.0
    gfs_press_drop_24h = gfs_press_min[idx - 24] - gfs_press if idx >= 24 else 0.0
    gfs_press_drop_3h  = gfs_press_min[idx - 3] - gfs_press if idx >= 3 else 0.0
    
    multiplier = 0.8 if (center_dir >= 315) or (center_dir <= 45) else (1.05 if 90 <= center_dir <= 180 else 1.0)
        
    ec_winds_adj = ec_winds * multiplier
    gfs_winds_adj = gfs_winds * multiplier

    # 物理與 AI 計算 (加入防護)
    ec_t1_phy = (ec_press <= 1005) & (ec_winds_adj >= 38) & (ec_press_drop_24h >= 1.5)
    gfs_t1_phy = (gfs_press <= 1005) & (gfs_winds_adj >= 38) & (gfs_press_drop_24h >= 1.5)
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds_adj >= 48)) | (ec_winds_adj >= 55) | ((ec_press_drop_3h >= 2.0) & (ec_winds_adj >= 45))
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds_adj >= 48)) | (gfs_winds_adj >= 55) | ((gfs_press_drop_3h >= 2.0) & (gfs_winds_adj >= 45))
    
    len_ec = len(ec_press) if len(ec_press) > 0 else 1
    len_gfs = len(gfs_press) if len(gfs_press) > 0 else 1
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len_ec*100)*0.6) + ((np.sum(gfs_t1_phy)/len_gfs*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len_ec*100)*0.6) + ((np.sum(gfs_t8_phy)/len_gfs*100)*0.4), 1)

    ec_ai_preds = np.array([ai_model.predict([[ec_press[m], ec_press_drop_24h[m], ec_press_drop_3h[m], ec_winds_adj[m]]])[0] for m in range(len(ec_press))])
    gfs_ai_preds = np.array([ai_model.predict([[gfs_press[m], gfs_press_drop_24h[m], gfs_press_drop_3h[m], gfs_winds_adj[m]]])[0] for m in range(len(gfs_press))])
    
    prob_t1_ec_ai = (np.sum(ec_ai_preds >= 1) / len(ec_ai_preds) * 100) if len(ec_ai_preds) > 0 else 0.0
    prob_t8_ec_ai = (np.sum(ec_ai_preds == 8) / len(ec_ai_preds) * 100) if len(ec_ai_preds) > 0 else 0.0
    prob_t1_gfs_ai = (np.sum(gfs_ai_preds >= 1) / len(gfs_ai_preds) * 100) if len(gfs_ai_preds) > 0 else 0.0
    prob_t8_gfs_ai = (np.sum(gfs_ai_preds == 8) / len(gfs_ai_preds) * 100) if len(gfs_ai_preds) > 0 else 0.0
    
    prob_t1_ai = round((prob_t1_ec_ai * 0.6) + (prob_t1_gfs_ai * 0.4), 1)
    prob_t8_ai = round((prob_t8_ec_ai * 0.6) + (prob_t8_gfs_ai * 0.4), 1)
    model_spread = round((np.std(ec_winds_adj) * 0.6) + (np.std(gfs_winds_adj) * 0.4), 1) if len(ec_winds_adj) > 0 else 0

    results.append({
        "時間": display_time, "物理一號機率 (%)": prob_t1_phy, "物理八號機率 (%)": prob_t8_phy,
        "AI 一號機率 (%)": prob_t1_ai, "AI 八號機率 (%)": prob_t8_ai, "陣風分歧度 (Uncertainty)": model_spread
    })

df_res = pd.DataFrame(results)
df_res_filtered = df_res.iloc[::6, :].reset_index(drop=True)

# ==========================================
# 🗺️ 升級：AI 空間推算演算法 (動態生成路徑)
# ==========================================
def generate_ai_path(times_arr, winds_arr, dirs_dict):
    coords = []
    for i, t_str in enumerate(times_arr):
        ws = winds_arr[i]
        # 只在預測風力 >= 40km/h 時才視為有系統逼近並畫線
        if ws < 40: continue
        
        wd = dirs_dict.get(t_str, 90)
        # 根據白貝羅定律，風暴中心約在風向左前方 100 度
        storm_bearing = (wd - 100) % 360
        # 根據風速反向推算距離 (風越大，距離越近，範圍 50km - 600km)
        distance_km = max(50, 600 - (ws * 5.5))
        
        # 極坐標轉經緯度 (1度緯度 ≈ 111km)
        bearing_rad = math.radians(storm_bearing)
        delta_lat = (distance_km * math.cos(bearing_rad)) / 111.0
        delta_lon = (distance_km * math.sin(bearing_rad)) / (111.0 * math.cos(math.radians(HK_CENTER[0])))
        
        coords.append([round(HK_CENTER[1] + delta_lon, 2), round(HK_CENTER[0] + delta_lat, 2)])
    return coords

# 提取平均預測風速生成動態路徑
ec_mean_winds = np.mean(ec_winds_max, axis=1)
gfs_mean_winds = np.mean(gfs_winds_max, axis=1)

ai_paths_data = {"type": "FeatureCollection", "features": []}

ec_coords = generate_ai_path(times, ec_mean_winds, dir_dict)
if len(ec_coords) >= 2:
    ai_paths_data["features"].append({
        "type": "Feature", "properties": { "model": "ECMWF AIFS (Data 推算)", "color": "#a855f7" },
        "geometry": { "type": "LineString", "coordinates": ec_coords }
    })

gfs_coords = generate_ai_path(times, gfs_mean_winds, dir_dict)
if len(gfs_coords) >= 2:
    ai_paths_data["features"].append({
        "type": "Feature", "properties": { "model": "GFS 物理模式 (Data 推算)", "color": "#ef4444" },
        "geometry": { "type": "LineString", "coordinates": gfs_coords }
    })

os.makedirs("data", exist_ok=True)
with open("data/ai_paths.geojson", "w", encoding="utf-8") as f:
    json.dump(ai_paths_data, f, ensure_ascii=False, indent=2)
geojson_json_str = json.dumps(ai_paths_data, ensure_ascii=False)

# ==========================================
# 📊 繪製終極對比圖表 (加入節點 + 固定 0-100% 比例)
# ==========================================
fig = go.Figure()
hover_temp = "%{y}%<br>模型分歧度: %{customdata} km/h"

fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理一號機率 (%)"], name="🟡 傳統物理 (一號風球)", mode='lines+markers', marker=dict(size=6), line=dict(color='yellow', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 一號機率 (%)"], name="🌟 AI 決策樹 (一號風球)", mode='lines+markers', marker=dict(size=7), line=dict(color='gold', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["物理八號機率 (%)"], name="🔵 傳統物理 (八號風球)", mode='lines+markers', marker=dict(size=6), line=dict(color='deepskyblue', width=2, dash='dash'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["AI 八號機率 (%)"], name="🔴 AI 決策樹 (八號風球)", mode='lines+markers', marker=dict(size=7), line=dict(color='red', width=3), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))

fig.update_layout(title="🌀 香港風暴信號預測：四維地形修正 AI 雙軌制", yaxis_title="發出機率 (%)", xaxis_title="預測時間", yaxis=dict(range=[0, 100]), hovermode="x unified", template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
table_html = df_res_filtered.to_html(index=False, border=0)
chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# ==========================================
# 🧠 AI 自動生成綜合結論 (防彈安全版)
# ==========================================
downgrade_text = ""
if df_res_filtered.empty or df_res_filtered["AI 八號機率 (%)"].isna().all():
    peak_time, peak_prob, peak_spread = "暫無數據", 0.0, 0.0
else:
    df_res_filtered["AI 八號機率 (%)"] = df_res_filtered["AI 八號機率 (%)"].fillna(0)
    max_t8_idx = int(df_res_filtered["AI 八號機率 (%)"].idxmax())
    max_t8_row = df_res_filtered.iloc[max_t8_idx]
    peak_time, peak_prob, peak_spread = max_t8_row["時間"], max_t8_row["AI 八號機率 (%)"], max_t8_row["陣風分歧度 (Uncertainty)"]

    if peak_prob >= 20.0:
        after_peak_df = df_res_filtered.iloc[max_t8_idx + 1:]
        downgrade_candidates = after_peak_df[after_peak_df["AI 八號機率 (%)"] < 20.0]
        if not downgrade_candidates.empty:
            dr = downgrade_candidates.iloc[0]
            downgrade_text = f"<br><br>📉 <b>( 8 ➡️ 3) 最有落波時間為香港時間【{dr['時間']}】，機率為 {dr['AI 八號機率 (%)']}%</b> <span style='color:#ffaaaa;'>(陣風分歧度：{dr['陣風分歧度 (Uncertainty)']} km/h)</span>。"
        else:
            downgrade_text = "<br><br>📉 <b>( 8 ➡️ 3) 落波評估：</b>風暴影響時間較長，預測期結束前暫未見明確落波信號。"

if peak_prob >= 20.0:
    conclusion_html = f"""<div style="background: linear-gradient(135deg, #4b1313, #8b0000); padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; border-left: 4px solid #ff3333; color: #fff;">🚨 <b>AI 實時威脅判定：</b><br>預計<b>最有可能懸掛八號風球的時間為【{peak_time}】</b>，最高機率達到 <b>{peak_prob}%</b> <span style="color:#ffaaaa;">(陣風分歧度：{peak_spread} km/h)</span>。{downgrade_text}</div>"""
elif peak_prob > 0:
    conclusion_html = f"""<div style="background: #2b2b00; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; border-left: 4px solid #ffd700; color: #fff;">⚠️ <b>AI 實時威脅判定：</b><br>系統偵測到八號風球信號，預計高峰期為<b>【{peak_time}】</b>，機率為 <b>{peak_prob}%</b>。目前威脅屬於中低度或處於分歧狀態，請密切留意。</div>"""
else:
    conclusion_html = f"""<div style="background: #1a2a1a; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 15px; border-left: 4px solid #33ff33; color: #fff;">✅ <b>AI 實時威脅判定：</b><br>根據當前數據計算，未來 10 天內<b>未偵測到實質的八號風球威脅</b>。地圖將保持清空狀態。</div>"""

hkt_now = datetime.now(timezone.utc) + timedelta(hours=8)
update_time_str = hkt_now.strftime('%Y-%m-%d %H:%M')

# ==========================================
# 🖥️ 終極 HTML 拼裝
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
        body {{ font-family: -apple-system, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 15px; }}
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
        #typhoon-map {{ width: 100%; height: 500px; border-radius: 8px; background: #1a1a1a; margin-top: 25px; border: 1px solid #333; }}
        .map-legend {{ background: rgba(30,30,30,0.85); color: #fff; padding: 10px; border-radius: 8px; font-size: 12px; border: 1px solid #444; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 14px; height: 4px; margin-right: 8px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌀 香港潛在風暴雙軌早期預警系統</h1>
            <div class="update-time">最後自動更新（HKT）: {update_time_str}</div>
        </div>
        {conclusion_html}
        <div class="intro-box">
            💡 <b>演算法升級 (Data-Driven AI Path)：</b> 系統已修復「大氣日夜潮汐」導致的假陽性誤判。地圖路徑現已全面改用 <b>AI 白貝羅定律演算法</b>，實時分析香港風向與風速，動態反向計算風暴中心坐標。如無風暴威脅（陣風 <40km/h），地圖將自動保持清空。
        </div>
        <div>{chart_html}</div>
        
        <div id="typhoon-map"></div>
        <div style="font-size: 12px; color: #888; text-align: center; margin-top: 5px;">*地圖路徑由本系統 AI 基於實時風向/風速動態推算。</div>
        
        <div class="table-container"><h3>📋 四維綜合預測數據表</h3>{table_html}</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('typhoon-map').setView([21.2, 115.0], 7);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ maxZoom: 12, minZoom: 4 }}).addTo(map);

        const hkCoords = [22.3, 114.2];
        [
            {{ r: 200000, c: '#eab308', label: '200km 外圍烈風圈' }},
            {{ r: 100000, c: '#f97316', label: '100km 暴風核心圈' }},
            {{ r: 50000,  c: '#ef4444', label: '50km 穿心直擊圈' }}
        ].forEach(ring => L.circle(hkCoords, {{ color: ring.c, fillColor: ring.c, fillOpacity: 0.08, weight: 1.5, dashArray: '4,6' }}).bindTooltip(ring.label).addTo(map));
        L.circleMarker(hkCoords, {{ radius: 6, color: '#3b82f6', fillColor: '#60a5fa', fillOpacity: 1 }}).addTo(map).bindPopup('<b>香港 (Hong Kong)</b>');

        const geojsonData = {geojson_json_str};
        if (geojsonData.features.length > 0) {{
            L.geoJSON(geojsonData, {{
                style: f => ({{ color: f.properties.color, weight: 3.5, opacity: 0.85 }}),
                onEachFeature: (f, l) => l.bindPopup('<b>' + f.properties.model + '</b>')
            }}).addTo(map);
        }}

        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function () {{
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <div style="font-weight:bold; margin-bottom:6px;">🤖 AI 動態推算路徑</div>
                <div class="legend-item"><div class="legend-color" style="background:#a855f7;"></div>ECMWF 數據源推算</div>
                <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>GFS 數據源推算</div>
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

print("🎉 恭喜！包含 Data-Driven AI 地圖及修正時間蟲嘅終極版 index.html 已成功生成！")
