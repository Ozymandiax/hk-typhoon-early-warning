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
# 📍 香港及五星陣座標設定
# ==========================================
HK_CENTER = (22.3193, 114.1694)
LATS = "22.18,22.30,22.31,22.35,22.50"
LONS = "114.10,114.17,113.92,114.35,114.15"

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
print("✨ AI 訓練完成！")

# ==========================================
# ⚡ 獲取香港本地預測數據
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
num_times, num_ec_members, num_gfs_members = len(times), len(ec_wind_keys), len(gfs_wind_keys)

ec_winds_max = np.zeros((num_times, num_ec_members))
ec_press_min = np.full((num_times, num_ec_members), 1050.0)
gfs_winds_max = np.zeros((num_times, num_gfs_members))
gfs_press_min = np.full((num_times, num_gfs_members), 1050.0)

for loc_data in res_ec_list:
    h = loc_data['hourly']
    ec_winds_max = np.maximum(ec_winds_max, np.array([[h[k][i] or 0 for k in ec_wind_keys] for i in range(num_times)]))
    ec_press_min = np.minimum(ec_press_min, np.array([[h[k][i] or 1050.0 for k in ec_press_keys] for i in range(num_times)]))
for loc_data in res_gfs_list:
    h = loc_data['hourly']
    gfs_winds_max = np.maximum(gfs_winds_max, np.array([[h[k][i] or 0 for k in gfs_wind_keys] for i in range(num_times)]))
    gfs_press_min = np.minimum(gfs_press_min, np.array([[h[k][i] or 1050.0 for k in gfs_press_keys] for i in range(num_times)]))

# ==========================================
# 核心風球計算
# ==========================================
results = []
for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
    ec_winds, ec_press, gfs_winds, gfs_press = ec_winds_max[idx], ec_press_min[idx], gfs_winds_max[idx], gfs_press_min[idx]
    
    multiplier = 0.8 if (dir_dict.get(t_str, 90) >= 315 or dir_dict.get(t_str, 90) <= 45) else (1.05 if 90 <= dir_dict.get(t_str, 90) <= 180 else 1.0)
    ec_winds_adj, gfs_winds_adj = ec_winds * multiplier, gfs_winds * multiplier
    ec_p_drop = ec_press_min[max(0, idx - 24)] - ec_press
    gfs_p_drop = gfs_press_min[max(0, idx - 24)] - gfs_press

    ec_ai_preds = np.array([ai_model.predict([[ec_press[m], ec_p_drop[m], 0, ec_winds_adj[m]]])[0] for m in range(len(ec_press))])
    gfs_ai_preds = np.array([ai_model.predict([[gfs_press[m], gfs_p_drop[m], 0, gfs_winds_adj[m]]])[0] for m in range(len(gfs_press))])
    
    results.append({
        "時間": dt.strftime("%m月%d日 %H:00"),
        "物理一號機率 (%)": round(((np.sum(ec_winds_adj >= 38)/len(ec_press)*100)*0.6) + ((np.sum(gfs_winds_adj >= 38)/len(gfs_press)*100)*0.4), 1),
        "物理八號機率 (%)": round(((np.sum(ec_winds_adj >= 55)/len(ec_press)*100)*0.6) + ((np.sum(gfs_winds_adj >= 55)/len(gfs_press)*100)*0.4), 1),
        "AI 一號機率 (%)": round((np.mean(ec_ai_preds >= 1)*100 * 0.6) + (np.mean(gfs_ai_preds >= 1)*100 * 0.4), 1),
        "AI 八號機率 (%)": round((np.mean(ec_ai_preds == 8)*100 * 0.6) + (np.mean(gfs_ai_preds == 8)*100 * 0.4), 1),
        "陣風分歧度 (Uncertainty)": round((np.std(ec_winds_adj) * 0.6) + (np.std(gfs_winds_adj) * 0.4), 1)
    })

df_res_filtered = pd.DataFrame(results).iloc[::6, :].reset_index(drop=True)
os.makedirs("data", exist_ok=True)
df_res_filtered.to_excel("data/typhoon_predictions.xlsx", index=False, engine='openpyxl')

# ==========================================
# 🗺️ 終極版：100點高密度網格 + 質心平滑追蹤器
# ==========================================
print("🗺️ 正在使用 100 點高密度網格及『質心平滑演算法』追蹤氣旋...")

def classify_tc(ws):
    if ws < 41: return "熱帶低氣壓", "#3b82f6"
    elif ws < 63: return "熱帶風暴", "#06b6d4"
    elif ws < 88: return "強烈熱帶風暴", "#eab308"
    elif ws < 118: return "颱風", "#f97316"
    elif ws < 150: return "強颱風", "#ef4444"
    else: return "超強颱風", "#a855f7"

# 生成 10x10 完美矩陣 (覆蓋廣東、南海、台灣海峽)
lat_axes = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
lon_axes = [110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0]
grid_lats = [str(lat) for lat in lat_axes for _ in lon_axes]
grid_lons = [str(lon) for _ in lat_axes for lon in lon_axes]
url_grid = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(grid_lats)}&longitude={','.join(grid_lons)}&hourly=pressure_msl,wind_speed_10m&models=ecmwf_ifs025&forecast_days=7"

def fetch_tc_track():
    features = []
    try:
        res = requests.get(url_grid, timeout=20).json()
        res = res if isinstance(res, list) else [res]
        time_steps = len(res[0]['hourly']['time'])
        coords, last_lat, last_lon = [], None, None
        
        for t_idx in range(0, time_steps, 3):
            t_str = res[0]['hourly']['time'][t_idx]
            dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M") + timedelta(hours=8)
            time_disp = dt.strftime("%m-%d %H:00")
            
            # 1. 搵出全網格絕對最低氣壓
            min_p = min([pt['hourly']['pressure_msl'][t_idx] for pt in res if pt['hourly']['pressure_msl'][t_idx] is not None], default=1050.0)
            
            # 門檻：只有 <= 1012 hPa 才視為顯著低壓區開始畫線
            if min_p > 1012.0: 
                continue
                
            # 2. 搵出所有接近最低氣壓嘅區域 (誤差 1.0 內)
            valid_pts = [pt for pt in res if pt['hourly']['pressure_msl'][t_idx] is not None and pt['hourly']['pressure_msl'][t_idx] <= min_p + 1.0]
            
            if valid_pts:
                # 3. 質心演算法：計算氣壓谷底的平均經緯度（徹底解決卡死單點問題）
                avg_lat = sum(p['latitude'] for p in valid_pts) / len(valid_pts)
                avg_lon = sum(p['longitude'] for p in valid_pts) / len(valid_pts)
                max_ws = max((p['hourly']['wind_speed_10m'][t_idx] or 0) for p in valid_pts)
                
                # 防跳躍機制：如果3小時內瞬移超過300km，即係系統已經消失，斷開連線
                if last_lat is not None and math.hypot(avg_lat - last_lat, avg_lon - last_lon) * 111 > 300:
                    continue
                    
                last_lat, last_lon = avg_lat, avg_lon
                best_lat, best_lon = round(avg_lat, 2), round(avg_lon, 2)
                coords.append([best_lon, best_lat])
                
                lat1, lon1, lat2, lon2 = map(math.radians, [HK_CENTER[0], HK_CENTER[1], best_lat, best_lon])
                dist_hk = round(6371 * 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2)))
                cat, color = classify_tc(max_ws)
                
                features.append({
                    "type": "Feature",
                    "properties": { "time": time_disp, "model": "ECMWF 質心追蹤", "wind_speed": f"{int(max_ws)} km/h", "pressure": f"{round(min_p, 1)} hPa", "category": cat, "color": color, "dist_hk": f"{dist_hk} km", "position": f"{best_lat}°N, {best_lon}°E", "type": "track_point" },
                    "geometry": { "type": "Point", "coordinates": [best_lon, best_lat] }
                })
                
        # 只要收集到最少 2 個連續質心，就畫出路徑折線！
        if len(coords) >= 2:
            features.insert(0, {
                "type": "Feature",
                "properties": { "model": "預測軌跡連線", "color": "#00f2fe", "type": "track_line" },
                "geometry": { "type": "LineString", "coordinates": coords }
            })
    except Exception as e:
        print(f"⚠️ 網格追蹤分析出錯: {e}")
    return features

ai_paths_data = {"type": "FeatureCollection", "features": fetch_tc_track()}
with open("data/ai_paths.geojson", "w", encoding="utf-8") as f:
    json.dump(ai_paths_data, f, ensure_ascii=False, indent=2)

geojson_json_str = json.dumps(ai_paths_data, ensure_ascii=False)

# ==========================================
# 📊 Plotly 圖表 + HTML 拼裝
# ==========================================
fig = go.Figure()
hover_temp = "%{y}%<br>模型分歧度: %{customdata} km/h"
for col, name, colr in [("物理一號機率 (%)", "🟡 傳統物理 (一號)", "yellow"), ("AI 一號機率 (%)", "🌟 AI 決策樹 (一號)", "gold"), ("物理八號機率 (%)", "🔵 傳統物理 (八號)", "deepskyblue"), ("AI 八號機率 (%)", "🔴 AI 決策樹 (八號)", "red")]:
    fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered[col], name=name, mode='lines+markers', line=dict(color=colr, width=2 if '物理' in name else 3, dash='dash' if '物理' in name else 'solid'), customdata=df_res_filtered["陣風分歧度 (Uncertainty)"], hovertemplate=hover_temp))

fig.update_layout(title="🌀 香港風暴信號預測", yaxis=dict(range=[0, 100]), template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")

hkt_now = datetime.now(timezone.utc) + timedelta(hours=8)
update_time_str = hkt_now.strftime('%Y-%m-%d %H:%M')

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
        h1 {{ text-align: center; color: #ff5252; font-size: 24px; padding-bottom:10px; border-bottom: 1px solid #1c2541;}}
        .update-time {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 20px; }}
        #typhoon-map {{ width: 100%; height: 550px; border-radius: 10px; background: #0b132b; margin: 20px 0; border: 1px solid #3a506b; }}
        .tc-legend {{ background: rgba(11,19,43,0.9); color: #fff; padding: 12px; border-radius: 8px; font-size: 12px; border: 1px solid #3a506b; line-height: 1.8; }}
        .legend-badge {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌀 香港潛在風暴預警與氣旋質心追蹤系統</h1>
        <div class="update-time">最後更新（HKT）: {update_time_str}</div>
        
        <div style="background: #1c2541; padding: 15px; border-radius: 6px; font-size: 14px; line-height: 1.6; border-left: 4px solid #4facfe;">
            💡 <b>地圖追蹤升級：</b> 已啟用「10x10 高密度海洋網格」並套用「質心平滑演算法 (Center of Mass)」。系統會自動過濾雜訊，精確連線低壓槽與熱帶氣旋的平滑移動軌跡，拒絕卡頓跳躍！
        </div>

        <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div id="typhoon-map"></div>
        <div style="overflow-x: auto; background: #1c2541; padding: 15px; border-radius: 8px; margin-top: 20px;">
            {df_res_filtered.to_html(index=False, border=0)}
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('typhoon-map').setView([20.0, 116.5], 6);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ maxZoom: 12, minZoom: 4 }}).addTo(map);

        const hkCoords = [22.3193, 114.1694];
        L.circleMarker(hkCoords, {{ radius: 7, color: '#00f2fe', fillColor: '#4facfe', fillOpacity: 1 }}).addTo(map).bindPopup('<b>香港 (Hong Kong)</b>');
        
        // 警戒圈
        [200000, 400000, 600000].forEach((r, idx) => {{
            L.circle(hkCoords, {{ radius: r, color: ['#ef4444', '#f97316', '#eab308'][idx], weight: 1.2, dashArray: '6, 8', fill: false, opacity: 0.6 }}).addTo(map);
        }});

        const geojsonData = {geojson_json_str};
        if (geojsonData.features.length > 0) {{
            L.geoJSON(geojsonData, {{
                style: function(f) {{
                    if (f.properties.type === "track_line") return {{ color: f.properties.color, weight: 3.5, opacity: 0.9, dashArray: '6, 6' }};
                }},
                pointToLayer: function(f, latlng) {{
                    if (f.properties.type === "track_point") {{
                        return L.circleMarker(latlng, {{ radius: 5.5, fillColor: f.properties.color, color: '#000', weight: 1, opacity: 1, fillOpacity: 1 }});
                    }}
                }},
                onEachFeature: function(f, layer) {{
                    if (f.geometry.type === "Point") {{
                        const p = f.properties;
                        layer.bindPopup(
                            '<div style="font-size:13px; line-height:1.5;">' +
                            '<b style="color:' + p.color + ';">' + p.category + ' (' + p.wind_speed + ')</b><br>' +
                            '<b>時間：</b>' + p.time + '<br><b>氣壓：</b>' + p.pressure + '<br>' +
                            '<b>距港：</b>' + p.dist_hk + '<br><b>位置：</b>' + p.position + '</div>'
                        );
                    }}
                }}
            }}).addTo(map);
        }}

        const legend = L.control({{ position: 'topright' }});
        legend.onAdd = function () {{
            const div = L.DomUtil.create('div', 'tc-legend');
            div.innerHTML = `<b>🌀 氣旋等級</b><br><span class="legend-badge" style="background:#3b82f6;"></span>熱帶低氣壓<br><span class="legend-badge" style="background:#06b6d4;"></span>熱帶風暴<br><span class="legend-badge" style="background:#eab308;"></span>強烈熱帶風暴<br><span class="legend-badge" style="background:#f97316;"></span>颱風<br><span class="legend-badge" style="background:#ef4444;"></span>強颱風<br><span class="legend-badge" style="background:#a855f7;"></span>超強颱風`;
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("🎉 終極修復完成！請 Push 到 GitHub 測試。")
