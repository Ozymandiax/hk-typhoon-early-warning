import pandas as pd
import numpy as np
import xarray as xr
import geopy.distance
from ecmwf.opendata import Client
import plotly.graph_objects as go
import os

HK_COORDS = (22.3, 114.2)
target_file = "ecmwf_ens.grib2"

# 1. 下載數據
print("正在下載 ECMWF 數據...")
client = Client(source="ecmwf")
client.retrieve(
    time=0,             
    stream="enfo",      
    type="pf",          
    step=[i for i in range(0, 240, 12)],  
    param=["msl", "10u", "10v"],        
    target=target_file            
)

# 2. 追蹤低壓
print("正在追蹤氣旋胚胎...")
ds = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'msl'}})
ds_sub = ds.sel(latitude=slice(30, 5), longitude=slice(100, 150))

all_tracks = []
for member in ds_sub.number.values:
    member_data = ds_sub.sel(number=member)
    for step in ds_sub.step.values:
        step_data = member_data.sel(step=step)
        msl_grid = step_data.msl.values
        min_idx = np.unravel_index(np.argmin(msl_grid), msl_grid.shape)
        min_val = msl_grid[min_idx] / 100.0  
        lat = step_data.latitude.values[min_idx[0]]
        lon = step_data.longitude.values[min_idx[1]]
        
        if min_val < 1008:
            forecast_time = pd.to_datetime(step_data.valid_time.values)
            all_tracks.append({
                "member": int(member),
                "step": int(step / np.timedelta64(1, 'h')),
                "time": forecast_time,
                "lat": float(lat),
                "lon": float(lon)
            })

df_tracks = pd.DataFrame(all_tracks)

# 3. 香港威脅評估
print("正在評估香港威脅機率...")
if df_tracks.empty:
    # 建立無颱風時的 HTML
    html_content = "<html><body style='font-family:sans-serif; text-align:center; padding-top:100px;'><h1>🎉 未來 10 天內香港無明顯風暴威脅</h1></body></html>"
else:
    ds_u = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '10u'}})
    ds_v = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '10v'}})
    hk_wind_speed = np.sqrt(ds_u.u10**2 + ds_v.v10**2).sel(latitude=22.3, longitude=114.2, method='nearest')

    results = []
    unique_steps = df_tracks['step'].unique()
    for step in sorted(unique_steps):
        step_tracks = df_tracks[df_tracks['step'] == step]
        target_time = step_tracks['time'].iloc[0]
        
        in_800km = 0
        in_400km = 0
        for _, row in step_tracks.iterrows():
            dist = geopy.distance.geodesic((row['lat'], row['lon']), HK_COORDS).km
            if dist <= 800:
                in_800km += 1
            if dist <= 400:
                in_400km += 1
                
        hk_winds = hk_wind_speed.sel(step=np.timedelta64(step, 'h')).values * 3.6 
        strong_wind = np.sum(hk_winds >= 41)
        gale_wind = np.sum((hk_winds * 1.2) >= 63) 
        
        total = 50.0
        results.append({
            "時間": target_time.strftime("%m月%d日 %H:00"),
            "一號戒備信號 (%)": round((in_800km / total) * 100, 1),
            "三號強風信號 (%)": round((strong_wind / total) * 100, 1),
            "八號烈風或暴風信號 (%)": round((min(gale_wind, in_400km) / total) * 100, 1)
        })
        
    df_res = pd.DataFrame(results)

    # 4. 生成 Plotly 互動圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["一號戒備信號 (%)"], name="一號信號機率 (%)", line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["三號強風信號 (%)"], name="三號信號機率 (%)", line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["八號烈風或暴風信號 (%)"], name="八號信號機率 (%)", line=dict(color='red', width=3)))
    
    fig.update_layout(
        title="🌀 未來 10 天香港風暴信號掛牌機率預測",
        yaxis_title="機率 (%)",
        xaxis_title="預測時間段",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e"
    )

    # 5. 組裝完整的 HTML 網頁
    table_html = df_res.to_html(index=False, border=0)
    # 將 Plotly 圖表轉為 HTML 程式碼片段
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>香港潛在風暴早期預警系統</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            h1 {{ text-align: center; color: #ff5252; }}
            .update-time {{ text-align: center; color: #888; font-size: 14px; margin-bottom: 20px; }}
            .table-container {{ margin-top: 30px; overflow-x: auto; background: #1e1e1e; padding: 15px; border-radius: 8px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #333; }}
            th {{ background-color: #2b2b2b; color: #fff; }}
            tr:hover {{ background-color: #252525; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌀 香港潛在風暴早期預警系統 (Static 版)</h1>
            <div class="update-time">最後更新時間（UTC）: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
            <div>{chart_html}</div>
            <div class="table-container">
                <h3>📋 詳細數據預測表</h3>
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """

# 寫出為 index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("index.html 生成完畢！")