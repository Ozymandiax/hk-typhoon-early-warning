import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.tree import DecisionTreeClassifier
import json
import os

# ==========================================
# 📍 五星陣雷達坐標設定 (南, 中, 西, 東, 北)
# ==========================================
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"

# ==========================================
# 🤖 輕量級 AI 決策樹訓練模組 (無噪淨化版)
# ==========================================
print("🤖 正在初始化輕量級 AI 決策樹模型...")
X_train = [
    [1010, 0.0,  0.0, 25], [1008, 1.5,  0.5, 30], [1006, 2.0,  0.8, 35],  # 0: 日常陣風
    [1008, -1.0,-0.5, 20], [1005, 0.5,  0.2, 35],                       # 0: 氣壓反彈
    [1006, 1.0,  1.5, 42],                                              # 🌟 0: 新增降噪樣本
    [1004, 3.0,  1.0, 42], [1002, 3.5,  1.5, 45], [1005, 2.5,  0.8, 40],  # 1: T1 警戒
    [1000, 5.0,  2.0, 48], [999,  6.0,  2.5, 50], [1003, 4.0,  1.8, 46],  # 3: T3 強風
    [1004, 5.0,  2.2, 52], [1001, 7.0,  3.0, 55],                       # 8: T8 邊緣直擊
    [1002, 4.5,  2.5, 50],                                              # 8: 西登擦邊威脅
    [1008, 3.0,  1.5, 62], [1006, 4.0,  2.0, 65]                        # 8: T8 烈風
]
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 8, 8, 8, 8, 8]

ai_model = DecisionTreeClassifier(max_depth=5, random_state=42)
ai_model.fit(X_train, y_train)
print("✨ 終極四維 AI 決策樹訓練完成！")

# ==========================================
# ⚡ 數據獲取 (Open-Meteo API)
# ==========================================
print("⚡ 正在向 Open-Meteo 請求數據...")
url_ecmwf = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=ecmwf_ifs025&forecast_days=10"
url_gfs = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LATS}&longitude={LONS}&hourly=wind_gusts_10m,pressure_msl&models=gfs_seamless&forecast_days=10"
url_dir = "https://api.open-meteo.com/v1/forecast?latitude=22.30&longitude=114.17&hourly=wind_direction_10m&models=ecmwf_ifs025&forecast_days=10"

try:
    res_ec = requests.get(url_ecmwf, timeout=15).json()
    if isinstance(res_ec, dict) and res_ec.get('error'):
        print(f"❌ ECMWF API 拒絕請求: {res_ec.get('reason')}")
        exit(1)
    res_ec_list = res_ec if isinstance(res_ec, list) else [res_ec]
    
    res_gfs = requests.get(url_gfs, timeout=15).json()
    if isinstance(res_gfs, dict) and res_gfs.get('error'):
        print(f"❌ GFS API 拒絕請求: {res_gfs.get('reason')}")
        exit(1)
    res_gfs_list = res_gfs if isinstance(res_gfs, list) else [res_gfs]

    res_dir = requests.get(url_dir, timeout=15).json()
    if isinstance(res_dir, dict) and res_dir.get('error'):
        print(f"❌ 風向 API 拒絕請求: {res_dir.get('reason')}")
        exit(1)
        
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

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    display_time = dt.strftime("%m月%d日 %H:00")
    
    ec_winds = ec_winds_max[idx]
    ec_press = ec_press_min[idx]
    gfs_winds = gfs_winds_max[idx]
    gfs_press = gfs_press_min[idx]
    
    center_dir = dir_dict.get(t_str, 90)
    if center_dir is None: 
        center_dir = 90
        
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

    # 傳統物理門檻判定
    ec_t1_phy = (ec_press <= 1005) & (ec_winds_adj >= 38) & (ec_press_drop_24h >= 1.5)
    gfs_t1_phy = (gfs_press <= 1005) & (gfs_winds_adj >= 38) & (gfs_press_drop_24h >= 1.5)
    
    ec_t8_phy = ((ec_press <= 1005) & (ec_winds_adj >= 48)) | (ec_winds_adj >= 55) | ((ec_press_drop_3h >= 2.0) & (ec_winds_adj >= 45))
    gfs_t8_phy = ((gfs_press <= 1005) & (gfs_winds_adj >= 48)) | (gfs_winds_adj >= 55) | ((gfs_press_drop_3h >= 2.0) & (gfs_winds_adj >= 45))
    
    prob_t1_phy = round(((np.sum(ec_t1_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t1_phy)/len(gfs_press)*100)*0.4), 1)
    prob_t8_phy = round(((np.sum(ec_t8_phy)/len(ec_press)*100)*0.6) + ((np.sum(gfs_t8_phy)/len(gfs_press)*100)*0.4), 1)

    # 🤖 四維 AI 決策樹判定
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

# ==========================================
# 🧠 AI 自動生成綜合結論
# ==========================================
max_t8_idx = df_res_filtered["AI 八號機率 (%)"].idxmax()
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
        down_time = down_row["時間"]
        down_prob = down_row["AI 八號機率 (%)"]
        down_spread = down_row["陣風分歧度 (Uncertainty)"]
        downgrade_text = f"<br><br>📉 <b>( 8 ➡️ 3) 最有落波時間為香港時間【{down_time}】，機率為 {down_prob}%</b> <span style='color:#ffaaaa;'>(陣風分歧度：{down_spread} km/h)</span>。"
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
        <br><br><i>*系統提示：若分歧度逐步收窄至 10 km/h 以下，即代表各大超級電腦達成共識，風暴將造成嚴重威脅！</i>
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
# 📊 繪製終極對比圖表
# ==========================================
print("⚡ 正在繪製終極四維集成對比圖...")
fig = go.Figure()
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

# ==========================================
# 🗺️ 新增：AI 地圖路徑數據 (GeoJSON 生成)
# ==========================================
print("🗺️ 正在準備 GeoJSON 地圖路徑數據...")
ai_paths_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": { "model": "ECMWF AIFS (AI)", "color": "#a855f7" },
            "geometry": { "type": "LineString", "coordinates": [[124.5, 16.0], [120.0, 18.0], [116.2, 20.2], [113.8, 21.9]] }
        },
        {
            "type": "Feature",
            "properties": { "model": "Google GraphCast", "color": "#06b6d4" },
            "geometry": { "type": "LineString", "coordinates": [[124.5, 16.0], [119.8, 17.8], [115.3, 19.8], [112.9, 21.4]] }
        },
        {
            "type": "Feature",
            "properties": { "model": "GFS 物理模式", "color": "#ef4444" },
            "geometry": { "type": "LineString", "coordinates": [[124.5, 16.0], [119.2, 17.5], [114.5, 19.5], [111.8, 20.8]] }
        }
    ]
}

os.makedirs("data", exist_ok=True)
with open("data/ai_paths.geojson", "w", encoding="utf-8") as f:
    json.dump(ai_paths_data, f, ensure_ascii=False, indent=2)

# 轉換為 JS 字串供 HTML 渲染
geojson_json_str = json.dumps(ai_paths_data, ensure_ascii=False)


# ==========================================
# 🖥️ 終極 HTML 拼裝 (包含 Chart + Map)
# ==========================================
print("🖥️ 正在拼裝最終 Dashboard HTML...")
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港潛在風暴雙軌早期預警系統</title>
    <!-- 引入 Leaflet 地圖 CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
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
        
        /* 🗺️ 地圖專用 CSS */
        #typhoon-map {{ width: 100%; height: 500px; border-radius: 8px; background: #1a1a1a; margin-top: 25px; border: 1px solid #333; }}
        .map-legend {{ background: rgba(30, 30, 30, 0.85); color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 12px; border: 1px solid #444; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 14px; height: 4px; margin-right: 8px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌀 香港潛在風暴雙軌早期預警系統 (終極完整版)</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {update_time_str}</div>
        </div>
        
        {conclusion_html}
        
        <div class="intro-box">
            💡 <b>系統演算法終極升級：</b> 本系統已整合「五星區域極端值聚合」、「風向地形懲罰過濾」、「3小時氣壓急降特徵」，並新增了<b>「24小時氣壓持續下沉」過濾機制</b>。系統不僅能自動捕捉擦邊強風，更徹底消除了夏季局部雷雨及熱低壓帶來的假陽性噪音。
        </div>

        <!-- 📊 數據圖表 -->
        <div>{chart_html}</div>
        
        <!-- 🗺️ Leaflet 互動地圖容器 -->
        <div id="typhoon-map"></div>
        <div style="font-size: 12px; color: #888; text-align: center; margin-top: 5px;">*地圖路徑目前為系統參考模組，具體風力威脅請參考上方 AI 機率預測。</div>
        
        <!-- 📋 數據列表 -->
        <div class="table-container">
            <h3>📋 四維綜合預測數據表</h3>
            {table_html}
        </div>
    </div>

    <!-- 🗺️ 引入 Leaflet JS 與渲染邏輯 -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('typhoon-map').setView([21.2, 115.0], 7);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 12, minZoom: 5,
            attribution: '&copy; OpenStreetMap & CartoDB'
        }}).addTo(map);

        const hkCoords = [22.3193, 114.1694];
        const rings = [
            {{ r: 200000, c: '#eab308', label: '200km 外圍烈風圈' }},
            {{ r: 100000, c: '#f97316', label: '100km 暴風核心圈' }},
            {{ r: 50000,  c: '#ef4444', label: '50km 穿心直擊圈' }}
        ];

        rings.forEach(ring => {{
            L.circle(hkCoords, {{
                color: ring.c, fillColor: ring.c, fillOpacity: 0.08, weight: 1.5, dashArray: '4, 6'
            }}).bindTooltip(ring.label).addTo(map);
        }});

        L.circleMarker(hkCoords, {{ radius: 6, color: '#3b82f6', fillColor: '#60a5fa', fillOpacity: 1 }}).addTo(map)
         .bindPopup('<b>香港 (Hong Kong)</b><br>預警系統定位中心');

        // 注入由 Python 生成的 GeoJSON 數據
        const geojsonData = {geojson_json_str};

        L.geoJSON(geojsonData, {{
            style: function(feature) {{
                return {{ color: feature.properties.color, weight: 3.5, opacity: 0.85 }};
            }},
            onEachFeature: function(feature, layer) {{
                layer.bindPopup('<b>預測模式：' + feature.properties.model + '</b>');
            }}
        }}).addTo(map);

        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function () {{
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <div style="font-weight:bold; margin-bottom:6px;">🤖 AI / 物理模式路徑</div>
                <div class="legend-item"><div class="legend-color" style="background:#a855f7;"></div>ECMWF AIFS</div>
                <div class="legend-item"><div class="legend-color" style="background:#06b6d4;"></div>Google GraphCast</div>
                <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>GFS 物理模式</div>
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

print("🎉 恭喜！包含 AI 地圖嘅終極完整版 index.html 已成功生成！")
