import pandas as pd
import numpy as np
import xarray as xr
import geopy.distance
from ecmwf.opendata import Client
import os
import gc

HK_COORDS = (22.3, 114.2)
target_file = "ecmwf_ens.grib2"

# 1. 下載最新數據
print("⚡ 開始下載最新 ECMWF 數據...")
try:
    client = Client(source="ecmwf")
    if os.path.exists(target_file):
        os.remove(target_file)
    client.retrieve(
        time=0,             
        stream="enfo",      
        type="pf",          
        step=[i for i in range(0, 240, 12)],  
        param=["msl", "10u", "10v"],        
        target=target_file            
    )
    print("✨ 數據下載成功！")
except Exception as e:
    print(f"❌ 下載失敗: {e}")
    exit(1)

# 2. 讀取氣壓數據（低記憶體優化模式）
print("⚡ 正在讀取氣壓數據並追蹤氣旋胚胎...")
ds = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'msl'}})
ds_sub = ds.sel(latitude=slice(30, 5), longitude=slice(100, 150))
ds_stacked = ds_sub.stack(grid=('latitude', 'longitude'))

msl_hpa = ds_stacked.msl / 100.0  
min_grid_idx = msl_hpa.argmin(dim='grid').values

steps = ds_sub.step.values / np.timedelta64(1, 'h')
members = ds_sub.number.values
times = ds_sub.valid_time.values  

lats_list = ds_sub.latitude.values
lons_list = ds_sub.longitude.values
grid_vals = ds_stacked.grid.values
msl_vals = msl_hpa.values  

del ds, ds_sub, ds_stacked, msl_hpa
gc.collect()  

all_tracks = []
for m_idx, member in enumerate(members):
    for s_idx, step in enumerate(steps):
        grid_idx = min_grid_idx[m_idx, s_idx]
        flat_grid_point = grid_vals[grid_idx] 
        lat = float(flat_grid_point[0])
        lon = float(flat_grid_point[1])
        min_val = float(msl_vals[m_idx, s_idx, grid_idx])
        
        if min_val < 1008:
            all_tracks.append({
                "member": int(member),
                "step": int(step),
                "time": pd.to_datetime(times[s_idx]),
                "lat": lat,
                "lon": lon
            })
            
df_tracks = pd.DataFrame(all_tracks)
del msl_vals
gc.collect()

# 3. 香港威脅評估與生成 HTML
print("⚡ 正在評估香港風暴信號機率並生成 HTML...")
if df_tracks.empty:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>香港潛在風暴早期預警系統</title>
        <style>
            body { font-family: sans-serif; background-color: #121212; color: #fff; text-align: center; padding-top: 100px; }
            h1 { color: #2ecc71; }
        </style>
    </head>
    <body>
        <h1>🎉 未來 10 天內香港無明顯風暴威脅</h1>
        <p>系統最後自動更新時間（UTC）: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + """</p>
    </body>
    </html>
    """
else:
    ds_u = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '10u'}})
    hk_u = ds_u.u10.sel(latitude=22.3, longitude=114.2, method='nearest').values
    del ds_u
    gc.collect()
    
    ds_v = xr.open_dataset(target_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '10v'}})
    hk_v = ds_v.v10.sel(latitude=22.3, longitude=114.2, method='nearest').values
    del ds_v
    gc.collect()
    
    hk_wind_speed_matrix = np.sqrt(hk_u**2 + hk_v**2) * 3.6  
    
    results = []
    unique_steps = df_tracks['step'].unique()
    
    lat_grid, lon_grid = np.meshgrid(lats_list, lons_list, indexing='ij')
    dx = (lon_grid - HK_COORDS[1]) * 102.0
    dy = (lats_list.reshape(-1, 1) - HK_COORDS[0]) * 111.0 
    dist_matrix = np.sqrt(dx**2 + dy**2)
    
    for step in sorted(unique_steps):
        step_tracks = df_tracks[df_tracks['step'] == step]
        target_time = step_tracks['time'].iloc[0]
        step_idx = np.where(steps == step)[0][0]
        
        in_800km = 0
        in_400km = 0
        
        for _, row in step_tracks.iterrows():
            lat_idx = np.abs(lats_list - row['lat']).argmin()
            lon_idx = np.abs(lons_list - row['lon']).argmin()
            dist = dist_matrix[lat_idx, lon_idx]
            
            if dist <= 800:
                in_800km += 1
            if dist <= 400:
                in_400km += 1
                
        hk_winds = hk_wind_speed_matrix[:, step_idx]
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
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["一號戒備信號 (%)"], name="一號信機率 (%)", line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["三號強風信號 (%)"], name="三號信機率 (%)", line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=df_res["時間"], y=df_res["八號烈風或暴風信號 (%)"], name="八號信機率 (%)", line=dict(color='red', width=3)))
    
    fig.update_layout(
        title="🌀 未來 10 天香港風暴信號掛牌機率預測",
        yaxis_title="機率 (%)",
        xaxis_title="預測時間段",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e"
    )
    
    table_html = df_res.to_html(index=False, border=0)
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>香港潛在風暴早期預警系統</title>
        <style>
            body {{ font-family: sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
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
            <h1>🌀 香港潛在風暴早期預警系統</h1>
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

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("🎉 index.html 已成功更新！")
