import os, re, math, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMapWithTime

# =======================
# USER DEFAULT SETTINGS
# =======================
EVENT_DATE = "2025-07-04"
START_TIME = "20:00:00"
END_TIME   = "23:30:00"

FIREWORKS_START = "22:20:00"
FIREWORKS_END   = "22:40:00"

# baseline window end & fit window end (recommended)
BASELINE_END = "22:10:00"
FIT_END      = "23:00:00"

# Your wind csv (UTC timestamps in DATE)
WIND_CSV = "wind_speed_2025.csv"

# Sensor files
SENSOR_GLOB = "Data_*.csv"
PM_COL_CANDIDATES = ["PM2.5_Env", "PM2.5_Std", "PM2.5"]

# Fireworks source (your 47°38'N, 122°20'W)
SRC_LAT, SRC_LON = 47.633333, -122.333333

# Your sensors (skip S4)
SENSOR_LATLON = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

OUT_DIR = "calibrated_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Heatmap domain/grid (meters around source)
HALF_WIDTH_M = 2200
GRID_N = 70
DT_MIN = 1  # 1-minute model step

# Heatmap point conversion
GRID_STRIDE = 2
KEEP_FRAC = 0.08
HEAT_RADIUS = 45

# =======================
# Helpers: coords
# =======================
def latlon_to_xy_m(lat, lon, lat0, lon0):
    R = 6371000.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R * dlon * math.cos(math.radians(lat0))  # east
    y = R * dlat                                 # north
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    R = 6371000.0
    lat0_rad = np.deg2rad(lat0)
    lat = lat0 + np.rad2deg(y / R)
    lon = lon0 + np.rad2deg(x / (R * np.cos(lat0_rad)))
    return lat, lon

# =======================
# Helpers: wind
# =======================
def wind_to_uv(ws_mps, wd_from_deg):
    """meteorological FROM deg -> (u_east, v_north)"""
    if np.isnan(ws_mps) or np.isnan(wd_from_deg):
        return 0.0, 0.0
    wd_to = (wd_from_deg + 180.0) % 360.0
    theta = math.radians(wd_to)
    v_n = ws_mps * math.cos(theta)
    u_e = ws_mps * math.sin(theta)
    return u_e, v_n

def read_wind_local_1min(csv_path, t0, t1, tz="America/Los_Angeles"):
    met = pd.read_csv(csv_path, low_memory=False)
    met["dt"] = (
        pd.to_datetime(met["DATE"], utc=True, errors="coerce")
        .dt.tz_convert(tz)
        .dt.tz_localize(None)
    )
    met["wind_speed"] = pd.to_numeric(met.get("wind_speed"), errors="coerce")
    met["wind_dir"]   = pd.to_numeric(met.get("wind_direction"), errors="coerce")
    met.loc[met["wind_dir"] == 999, "wind_dir"] = np.nan

    met = met[["dt", "wind_speed", "wind_dir"]].dropna(subset=["dt"]).sort_values("dt")

    # IMPORTANT: collapse duplicate timestamps
    met = met.groupby("dt", sort=True).mean(numeric_only=True)

    # resample
    met = met.resample("1min").ffill()

    # clip (with buffer for ffill stability)
    met = met.loc[t0 - pd.Timedelta(minutes=2) : t1 + pd.Timedelta(minutes=2)]
    return met

# =======================
# Helpers: sensor CSV
# =======================
def read_sensor_csv(path: str, anchor_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Counter" not in df.columns:
        raise ValueError(f"{path}: missing Counter column")

    pm_col = None
    for c in PM_COL_CANDIDATES:
        if c in df.columns:
            pm_col = c
            break
    if pm_col is None:
        raise ValueError(f"{path}: missing PM2.5 column; tried {PM_COL_CANDIDATES}")

    df["Counter"] = pd.to_numeric(df["Counter"], errors="coerce")
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
    df = df.dropna(subset=["Counter", pm_col]).copy()
    df["Counter"] = df["Counter"].astype(int)

    df["dt"] = anchor_start + pd.to_timedelta(df["Counter"], unit="s")
    df = df.rename(columns={pm_col: "pm25"})
    return df[["dt", "pm25"]].sort_values("dt")

def load_all_sensors_1min(t0, t1):
    series = {}
    for fp in sorted(glob.glob(SENSOR_GLOB)):
        base = os.path.basename(fp)
        m = re.search(r"Data_(\d+)\.csv$", base)
        if not m:
            continue
        snum = int(m.group(1))
        if snum == 4:
            continue
        sid = f"S{snum}"
        if sid not in SENSOR_LATLON:
            continue
        df = read_sensor_csv(fp, anchor_start=t0)
        df = df[(df["dt"] >= t0) & (df["dt"] <= t1)]
        s = df.set_index("dt")["pm25"].resample("1min").mean()
        series[sid] = s
    wide = pd.concat(series, axis=1).sort_index()
    return wide

# =======================
# Puff model: predict unit-Q at sensors
# =======================
def predict_unitQ_at_sensors(times, wind_u, wind_v, src_xy, sensor_xy,
                             sigma0_m, K_m2s, tau_s=None, release_mask=None, dt_s=60.0):
    """
    returns G(T, Ns): unit-Q predicted concentration at sensors over time
    release_mask[t]=True means emit a puff at that time
    """
    T = len(times)
    Ns = sensor_xy.shape[0]
    if release_mask is None:
        release_mask = np.ones(T, dtype=bool)

    puff_x, puff_y, puff_release_idx = [], [], []
    G = np.zeros((T, Ns), dtype=float)

    for t_idx in range(T):
        # release puff only when mask says so (e.g., fireworks window)
        if release_mask[t_idx]:
            puff_x.append(src_xy[0])
            puff_y.append(src_xy[1])
            puff_release_idx.append(t_idx)

        # advect existing puffs
        u = wind_u[t_idx]
        v = wind_v[t_idx]
        for k in range(len(puff_x)):
            puff_x[k] += u * dt_s
            puff_y[k] += v * dt_s

        # compute concentration at sensors
        for k in range(len(puff_x)):
            age_s = (t_idx - puff_release_idx[k]) * dt_s
            if age_s < 0:
                continue
            sigma2 = sigma0_m**2 + 2.0 * K_m2s * age_s
            if sigma2 < 1.0:
                sigma2 = 1.0

            dx = sensor_xy[:, 0] - puff_x[k]
            dy = sensor_xy[:, 1] - puff_y[k]
            r2 = dx*dx + dy*dy

            base = (1.0 / (2.0 * math.pi * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
            if tau_s is not None and tau_s > 0:
                base *= math.exp(-age_s / tau_s)

            G[t_idx, :] += base

    return G

def fit_Q_and_rmse(y, G, w=None):
    """Solve optimal Q for fixed (sigma0,K,tau) and return RMSE."""
    if w is None:
        w = np.ones_like(y)
    yv = y.reshape(-1)
    gv = G.reshape(-1)
    wv = w.reshape(-1)

    # Q* closed-form
    num = np.sum(wv * yv * gv)
    den = np.sum(wv * gv * gv) + 1e-12
    Q = num / den

    resid = yv - Q * gv
    rmse = np.sqrt(np.mean(resid**2))
    return Q, rmse

def grid_search(y, times, wind_u, wind_v, src_xy, sensor_xy, release_mask, w=None):
    # Coarse but robust parameter grid
    sigma0_list = [30, 60, 100, 150, 220]          # meters
    K_list      = [10, 30, 80, 150, 250]           # m^2/s
    tau_list    = [None, 1200, 2400, 7200]         # seconds

    best = {"rmse": 1e99}
    for sigma0 in sigma0_list:
        for K in K_list:
            for tau in tau_list:
                G = predict_unitQ_at_sensors(
                    times, wind_u, wind_v, src_xy, sensor_xy,
                    sigma0_m=sigma0, K_m2s=K, tau_s=tau,
                    release_mask=release_mask, dt_s=60.0
                )
                Q, rmse = fit_Q_and_rmse(y, G, w=w)
                if rmse < best["rmse"]:
                    best = {"sigma0": sigma0, "K": K, "tau": tau, "Q": Q, "rmse": rmse}
    return best

# =======================
# Model field for HTML
# =======================
def field_to_points(lat_grid, lon_grid, Z, stride=2, keep_frac=0.08):
    Z = np.asarray(Z)
    zmax = np.nanmax(Z)
    if not np.isfinite(zmax) or zmax <= 0:
        return []
    thresh = keep_frac * zmax
    pts = []
    for i in range(0, Z.shape[0], stride):
        for j in range(0, Z.shape[1], stride):
            val = Z[i, j]
            if val >= thresh:
                pts.append([float(lat_grid[i, j]), float(lon_grid[i, j]), float(val)])
    return pts

def simulate_frames(times, wind_u, wind_v, src_xy, X, Y, lat_grid, lon_grid,
                    sigma0, K, tau, Q, release_mask):
    puff_x, puff_y, puff_release_idx = [], [], []
    frames = []
    labels = []
    dt_s = 60.0

    for t_idx, t in enumerate(times):
        if release_mask[t_idx]:
            puff_x.append(src_xy[0])
            puff_y.append(src_xy[1])
            puff_release_idx.append(t_idx)

        # advect
        u = wind_u[t_idx]; v = wind_v[t_idx]
        for k in range(len(puff_x)):
            puff_x[k] += u * dt_s
            puff_y[k] += v * dt_s

        # field
        Z = np.zeros_like(X, dtype=float)
        for k in range(len(puff_x)):
            age_s = (t_idx - puff_release_idx[k]) * dt_s
            sigma2 = sigma0**2 + 2.0*K*age_s
            if sigma2 < 1.0:
                sigma2 = 1.0
            r2 = (X - puff_x[k])**2 + (Y - puff_y[k])**2
            base = (Q / (2.0*math.pi*sigma2)) * np.exp(-r2/(2.0*sigma2))
            if tau is not None and tau > 0:
                base *= math.exp(-age_s / tau)
            Z += base

        frames.append(field_to_points(lat_grid, lon_grid, Z, stride=GRID_STRIDE, keep_frac=KEEP_FRAC))
        labels.append(t.strftime("%H:%M"))

    return frames, labels

# =======================
# Main
# =======================
def main():
    t0 = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    t1 = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")
    fw0 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_START}")
    fw1 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_END}")
    t_baseline_end = pd.Timestamp(f"{EVENT_DATE} {BASELINE_END}")
    t_fit_end      = pd.Timestamp(f"{EVENT_DATE} {FIT_END}")

    # 1) Load sensors (1min)
    wide = load_all_sensors_1min(t0, t1)

    # 2) Load wind (1min)
    wind = read_wind_local_1min(WIND_CSV, t0, t1)

    # 3) Align time index (intersection)
    idx = wide.index.intersection(wind.index)
    wide = wide.loc[idx]
    wind = wind.loc[idx]

    # 4) Build time arrays for fitting window (t0..fit_end)
    fit_idx = wide.index[(wide.index >= t0) & (wide.index <= t_fit_end)]
    wide_fit = wide.loc[fit_idx]
    wind_fit = wind.loc[fit_idx]

    # 5) Baseline subtraction per sensor (median in [t0, baseline_end])
    base_idx = wide.index[(wide.index >= t0) & (wide.index <= t_baseline_end)]
    baselines = wide.loc[base_idx].median(axis=0, skipna=True)

    y = (wide_fit - baselines).fillna(0.0)

    # Optional: clip negative enhancement (helps stability)
    y = y.clip(lower=0.0)

    # 6) Convert sensors/source to local meters
    lat0, lon0 = SRC_LAT, SRC_LON
    src_xy = latlon_to_xy_m(SRC_LAT, SRC_LON, lat0, lon0)

    sensors = list(wide_fit.columns)
    sensor_xy = np.array([latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], lat0, lon0) for s in sensors])

    # 7) wind to u,v arrays
    wind_u = []
    wind_v = []
    for _, r in wind_fit.iterrows():
        u, v = wind_to_uv(float(r["wind_speed"]), float(r["wind_dir"]) if np.isfinite(r["wind_dir"]) else np.nan)
        wind_u.append(u)
        wind_v.append(v)
    wind_u = np.array(wind_u, dtype=float)
    wind_v = np.array(wind_v, dtype=float)

    times = wide_fit.index.to_pydatetime()

    # 8) release only during fireworks window (puffs emitted only 22:20–22:40)
    release_mask = np.array([(t >= fw0) and (t <= fw1) for t in wide_fit.index], dtype=bool)

    # 9) Grid search calibration
    y_arr = y.values  # (T, Ns)
    best = grid_search(y_arr, times, wind_u, wind_v, src_xy, sensor_xy, release_mask, w=None)

    print("=== BEST PARAMS (from grid search) ===")
    print(best)

    # 10) Build predicted series at sensors with best params (for plots)
    G_best = predict_unitQ_at_sensors(
        times, wind_u, wind_v, src_xy, sensor_xy,
        sigma0_m=best["sigma0"], K_m2s=best["K"], tau_s=best["tau"],
        release_mask=release_mask, dt_s=60.0
    )
    pred_enh = best["Q"] * G_best
    pred = pred_enh + baselines[sensors].values  # add baseline back

    pred_df = pd.DataFrame(pred, index=wide_fit.index, columns=sensors)
    obs_df = wide_fit.copy()

    # 11) Save timeseries comparison plot
    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.2*len(sensors)), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    for ax, s in zip(axes, sensors):
        ax.plot(obs_df.index, obs_df[s].values, label=f"{s} observed", linewidth=1.0)
        ax.plot(pred_df.index, pred_df[s].values, label=f"{s} model", linewidth=1.0)
        ax.axvspan(fw0, fw1, color="gray", alpha=0.15)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time (local)")
    fig.suptitle("Observed vs Calibrated Gaussian Puff Model (baseline-added)")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "calibration_timeseries.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved:", out_png)

    # 12) Make calibrated HTML plume (full window)
    # Prepare full-window time axis
    full_times = pd.date_range(t0, t1, freq=f"{DT_MIN}min")
    # align wind to full times
    wind_full = wind.loc[wind.index.intersection(full_times)].reindex(full_times).ffill()
    # u,v arrays
    u_full, v_full = [], []
    for _, r in wind_full.iterrows():
        u, v = wind_to_uv(float(r["wind_speed"]), float(r["wind_dir"]) if np.isfinite(r["wind_dir"]) else np.nan)
        u_full.append(u); v_full.append(v)
    u_full = np.array(u_full); v_full = np.array(v_full)

    # release mask for full window
    release_full = np.array([(t >= fw0) and (t <= fw1) for t in full_times], dtype=bool)

    # grid in meters
    xs = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    ys = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    lat_grid, lon_grid = xy_to_latlon(X, Y, lat0, lon0)

    frames, labels = simulate_frames(
        full_times.to_pydatetime(), u_full, v_full, src_xy, X, Y, lat_grid, lon_grid,
        sigma0=best["sigma0"], K=best["K"], tau=best["tau"], Q=best["Q"],
        release_mask=release_full
    )

    m = folium.Map(location=(SRC_LAT, SRC_LON), zoom_start=13, tiles="OpenStreetMap")

    # markers
    folium.Marker((SRC_LAT, SRC_LON), popup="Fireworks Source").add_to(m)
    for s, (slat, slon) in SENSOR_LATLON.items():
        folium.CircleMarker((slat, slon), radius=6, popup=s, fill=True).add_to(m)

    HeatMapWithTime(
        frames,
        index=labels,
        radius=HEAT_RADIUS,
        min_opacity=0.35,
        max_opacity=0.95,
        use_local_extrema=True,
        auto_play=False
    ).add_to(m)

    out_html = os.path.join(OUT_DIR, "calibrated_puff_plume_time.html")
    m.save(out_html)
    print("Saved:", out_html)
    print("View via:")
    print("  python -m http.server 8000")
    print(f"  http://localhost:8000/{OUT_DIR}/calibrated_puff_plume_time.html")

if __name__ == "__main__":
    main()
