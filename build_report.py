import json
import os

# ==========================================
# 1. 你原本/真實嘅數據抓取與計算邏輯 (Your Real Logic)
# ==========================================
def calculate_real_dashboard_data():
    """
    在這裡放入你原本的計算代碼：
    1. 抓取 ECMWF / GFS / HKO 實時數據
    2. 套用四維地形修正與 8 站門檻
    3. 經 AI 決策樹計算掛 8 機率與分歧度
    4. 提取各 AI 模式的經緯度路徑點 (Lat, Lon)
    """
    
    # --- 範例：假設這是你運算出來的真實動態結果 ---
    calculated_prob = 85.5                  # 你的動態機率 (e.g. real_prob)
    calculated_dispersion = 40.8            # 你的動態分歧度 (e.g. real_sigma)
    peak_time_str = "07月26日 08:00"         # 你的動態最高機率時間
    
    # 你的動態 AI 路徑 (轉換成 GeoJSON 格式: [Longitude, Latitude])
    real_ai_features = [
        {
            "type": "Feature",
            "properties": { "model": "ECMWF AIFS (AI)", "color": "#a855f7" },
            "geometry": {
                "type": "LineString",
                "coordinates": [[124.5, 16.0], [120.0, 18.0], [116.2, 20.2], [113.8, 21.9]] # 換成你抓到的 real lon/lat
            }
        },
        {
            "type": "Feature",
            "properties": { "model": "Google GraphCast", "color": "#06b6d4" },
            "geometry": {
                "type": "LineString",
                "coordinates": [[124.5, 16.0], [119.8, 17.8], [115.3, 19.8], [112.9, 21.4]]
            }
        },
        {
            "type": "Feature",
            "properties": { "model": "GFS 物理模式", "color": "#ef4444" },
            "geometry": {
                "type": "LineString",
                "coordinates": [[124.5, 16.0], [119.2, 17.5], [114.5, 19.5], [111.8, 20.8]]
            }
        }
    ]

    return {
        "prob": calculated_prob,
        "dispersion": calculated_dispersion,
        "peak_time": peak_time_str,
        "features": real_ai_features
    }


# ==========================================
# 2. 生成 Hugging Face / Web 用的 HTML 與 GeoJSON
# ==========================================
def build_report():
    # 執行真實運算
    data = calculate_real_dashboard_data()
    
    # 構建 GeoJSON 數據結構
    ai_paths_geojson = {
        "type": "FeatureCollection",
        "features": data["features"]
    }

    # 輸出 GeoJSON 到 data/ 資料夾
    os.makedirs("data", exist_ok=True)
    with open("data/ai_paths.geojson", "w", encoding="utf-8") as f:
        json.dump(ai_paths_geojson, f, ensure_ascii=False, indent=2)

    geojson_str = json.dumps(ai_paths_geojson, ensure_ascii=False)

    # 動態注入計算結果到 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港潛在風暴雙軌預警 Dashboard & AI 作戰地圖</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .card {{ background-color: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
        .badge-danger {{ background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }}
        #map {{ width: 100%; height: 520px; border-radius: 10px; background: #0f172a; }}
        .map-legend {{ background: rgba(15, 23, 42, 0.85); color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 12px; border: 1px solid #334155; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 14px; height: 4px; margin-right: 8px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🌀 香港潛在風暴雙軌早期預警系統</h2>
        
        <div class="card">
            <h3>📊 懸掛八號風球機率預測 (AI 決策樹 + 物理雙軌)</h3>
            <p>⏱️ 最高機率懸掛時間：<b>【{data['peak_time']}】</b></p>
            <p>🚨 最高機率：<span class="badge-danger">{data['prob']}%</span> (陣風分歧度：<b>{data['dispersion']} km/h</b>)</p>
            <p style="color: #94a3b8; font-size: 0.85em;">
                * 系統提示：分歧度代表超級電腦對「擦邊 8 號」與「穿心強颱」上限之角力。
            </p>
        </div>

        <div class="card">
            <h3>🗺️ AI 模式實時路徑 & 警戒圈作戰地圖</h3>
            <div id="map"></div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([21.2, 115.0], 7);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ maxZoom: 12, minZoom: 5 }}).addTo(map);

        const hkCoords = [22.3193, 114.1694];
        [
            {{ r: 200000, c: '#eab308', l: '200km 外圍烈風圈' }},
            {{ r: 100000, c: '#f97316', l: '100km 暴風核心圈' }},
            {{ r: 50000,  c: '#ef4444', l: '50km 穿心直擊圈' }}
        ].forEach(ring => {{
            L.circle(hkCoords, {{ color: ring.c, fillColor: ring.c, fillOpacity: 0.08, weight: 1.5, dashArray: '4,6' }}).bindTooltip(ring.l).addTo(map);
        }});

        L.circleMarker(hkCoords, {{ radius: 6, color: '#3b82f6', fillColor: '#60a5fa', fillOpacity: 1 }}).addTo(map).bindPopup('<b>香港 (Hong Kong)</b>');

        const geojsonData = {geojson_str};
        L.geoJSON(geojsonData, {{
            style: f => ({{ color: f.properties.color, weight: 3.5, opacity: 0.85 }}),
            onEachFeature: (f, l) => l.bindPopup('<b>預測模式：' + f.properties.model + '</b>')
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

    print("✅ 成功將真實運算結果與地圖渲染至 index.html！")

if __name__ == "__main__":
    build_report()
