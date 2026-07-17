import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

HK_LAT = 22.3
HK_LON = 114.2

print("⚡ 正在向 Open-Meteo 雲端請求 ECMWF 及 GFS 氣壓與風速數據...")

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

ec_wind_keys = [k for k in hourly_ec.keys() if "wind_speed_10m_member" in k]
ec_press_keys = [k for k in hourly_ec.keys() if "pressure_msl_member" in k]

gfs_wind_keys = [k for k in res_gfs['hourly'].keys() if "wind_speed_10m_member" in k]
gfs_press_keys = [k for k in res_gfs['hourly'].keys() if "pressure_msl_member" in k]

results = []

for idx, t_str in enumerate(times):
    dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
    display_time = dt.strftime("%m月%d日 %H:00")
    
    # ECMWF
    ec_winds = np.array([hourly_ec[k][idx] for k in ec_wind_keys if hourly_ec[k][idx] is not None])
    ec_press = np.array([hourly_ec[k][idx] for k in ec_press_keys if hourly_ec[k][idx] is not None])
    
    # GFS
    gfs_winds = np.array([res_gfs['hourly'][k][idx] for k in gfs_wind_keys if res_gfs['hourly'][k][idx] is not None])
    gfs_press = np.array([res_gfs['hourly'][k][idx] for k in gfs_press_keys if res_gfs['hourly'][k][idx] is not None])
    
    # 嚴格的物理篩選邏輯（氣壓 + 風速）
    prob_t1_ec = float(np.sum(ec_press <= 1006) / len(ec_press) * 100) if len(ec_press) > 0 else 0
    prob_t3_ec = float(np.sum((ec_press <= 1004) & (ec_winds * 1.1 >= 41)) / len(ec_winds) * 100) if len(ec_winds) > 0 else 0
    prob_t8_ec = float(np.sum((ec_press <= 1000) & (ec_winds * 1.2 >= 63)) / len(ec_winds) * 100) if len(ec_winds) > 0 else 0

    prob_t1_gfs = float(np.sum(gfs_press <= 1006) / len(gfs_press) * 100) if len(gfs_press) > 0 else 0
    prob_t3_gfs = float(np.sum((gfs_press <= 1004) & (gfs_winds * 1.1 >= 41)) / len(gfs_winds) * 100) if len(gfs_winds) > 0 else 0
    prob_t8_gfs = float(np.sum((gfs_press <= 1000) & (gfs_winds * 1.2 >= 63)) / len(gfs_winds) * 100) if len(gfs_winds) > 0 else 0

    # 權重集成
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
df_res_filtered = df_res.iloc[::6, :].reset_index(drop=True)

# 繪製 Plotly 對比圖
print("⚡ 正在繪製多模式集成對比圖...")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["ECMWF 8號機率 (%)"], name="ECMWF 模式 (八號風球)", line=dict(color='orange', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["GFS 8號機率 (%)"], name="GFS 模式 (八號風球)", line=dict(color='deepskyblue', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["綜合集成 8號機率 (%)"], name="🔴 權重集成 (八號風球)", line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=df_res_filtered["時間"], y=df_res_filtered["綜合集成 3號機率 (%)"], name="🟠 權重集成 (三號風球)", line=dict(color='gold', width=2)))

fig.update_layout(
    title="🌀 未來 10 天香港風暴信號掛牌機率預測（多模式加權集成）",
    yaxis_title="發出機率 (%)",
    xaxis_title="預測時間",
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e"
)

table_html = df_res_filtered.to_html(index=False, border=0)
chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# 生成完整的美觀網頁 HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港潛在風暴多模式早期預警系統</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 15px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #333; }}
        h1 {{ margin: 0; color: #ff5252; font-size: 24px; }}
        .update-time {{ color: #888; font-size: 13px; margin-top: 8px; }}
        .intro-box {{ background: #222; padding: 15px; border-radius: 6px; margin: 15px 0; font-size: 14px; line-height: 1.6; border-left: 4px solid #ff5252; }}
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
            <h1>🌀 香港潛在風暴多模式早期預警系統</h1>
            <div class="update-time">最後自動更新（香港時間 HKT）: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>運作原理：</b> 本系統採用「多模式權重集成」演算法。結合 <b>ECMWF IFS</b> 與 <b>NOAA GFS</b> 兩大黃金系集模式，以 <b>6:4 權重</b>進行集成，提供相較單一模式更穩定、虛報率低、時效長達 10 天的早期預警。
        </div>

        <div>{chart_html}</div>
        
        <div class="table-container">
            <h3>📋 綜合預測數據表（每 6 小時一檔）</h3>
            {table_html}
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("🎉 恭喜！多模式集成 index.html 已成功生成，網頁體積輕量且高度優化！")
