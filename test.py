#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centered Gaussian Puff + calibration + time heatmap
- plume center fixed at fireworks source (no drifting)
- wind speed/direction rotates + stretches the Gaussian (anisotropic), but center stays fixed
- emissions only during fireworks window
- after fireworks: diffusion + exponential decay only
- 20:00 baseline shown via sensor-IDW background (NOT forced to 0)
- uses ALL sensors S1..S6 (including S4) for calibration & background
- fixes wind duplicate timestamp issue (groupby + circular mean)

Run:
  python fireworks_centered_puff_calibrate_heatmap.py
View:
  cd calibrated_centered_outputs
  python -m http.server 8000
  http://localhost:8000/calibrated_centered_puff_heatmap_time.html
"""

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMapWithTime


# =========================
# USER SETTINGS (edit here)
# =========================

EVENT_DATE = "2025-07-04"
START_TIME = "20:00:00"
END_TIME   = "23:30:00"

FIREWORKS_START = "22:20:00"
FIREWORKS_END   = "22:40:00"

# Calibration windows
BASELINE_END = "22:10:00"   # baseline from [t0..BASELINE_END]
FIT_END      = "23:00:00"   # fit window [t0..FIT_END]

# Inputs
SENSOR_GLOB = "Data_*.csv"
WIND_CSV    = "wind_speed_2025.csv"

# Fireworks source (center fixed)
SRC_LAT, SRC_LON = 47.633333, -122.333333

# Sensor locations (edit S4 lat/lon if needed)
SENSOR_LATLON = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    "S4": (47.63629, -122.33998),  # <-- you can edit
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

PM_COL_CANDIDATES = ["PM2.5_Env", "PM2.5_Std", "PM2.5"]

# Wind time handling:
# If wind DATE is UTC (common for NOAA), keep True; if already local, set False.
WIND_TIME_IS_UTC = True
LOCAL_TZ = "America/Los_Angeles"

# Heat grid
HALF_WIDTH_M = 2500.0
GRID_N = 35  # 35x35=1225 points per frame

# Heatmap style
HEAT_RADIUS = 22
MIN_OPACITY = 0.25
MAX_OPACITY = 0.95

# Weight scaling
GAMMA = 0.75     # <1 makes low values more visible
WMIN  = 0.0      # keep 0.0 first to guarantee you SEE something
MAX_POINTS_PER_FRAME = 2500  # keep file size reasonable

# Puff physics knobs (USED)
DT_S = 60.0

WIND_STRETCH_L = 1500.0   # larger -> less extreme
STRETCH_CAP    = 6.0
U_CAP          = 6.0
K_PERP_SCALE   = 1.0

# Wind smoothing (reduce “sudden jumps”)
WIND_SMOOTH_MIN = 7

# Output
OUT_DIR = "calibrated_centered_outputs"

# If you “see nothing”, CMAX is usually the culprit.
# This version auto-picks CMAX from 97th percentile, then caps to [60..350].
CMAX_AUTO = True
CMAX_FIXED = 200.0


# =========================
# Geo conversions
# =========================

def latlon_to_xy_m(lat, lon, lat0, lon0):
    R = 6371000.0
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r)) * R
    y = (lat_r - lat0_r) * R
    return x, y

def xy_to_latlon_grid(X, Y, lat0, lon0):
    R = 6371000.0
    lat0_r = np.deg2rad(lat0)
    lon0_r = np.deg2rad(lon0)
    lat = lat0_r + (Y / R)
    lon = lon0_r + (X / (R * np.cos(0.5 * (lat + lat0_r))))
    return np.rad2deg(lat), np.rad2deg(lon)


# =========================
# Sensors
# =========================

def read_sensor_csv_1hz(path, anchor_start):
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Counter" not in df.columns:
        raise ValueError(f"{path}: missing Counter column.")

    pm_col = None
    for cand in PM_COL_CANDIDATES:
        if cand in df.columns:
            pm_col = cand
            break
    if pm_col is None:
        raise ValueError(f"{path}: cannot find PM col in {PM_COL_CANDIDATES}")

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
        sid = f"S{snum}"
        if sid not in SENSOR_LATLON:
            continue

        df = read_sensor_csv_1hz(fp, anchor_start=t0)
        df = df[(df["dt"] >= t0) & (df["dt"] <= t1)].copy()
        s = df.set_index("dt")["pm25"].resample("1min").mean()
        series[sid] = s

    if not series:
        raise SystemExit("No sensor files parsed. Make sure Data_*.csv is in this folder.")

    wide = pd.concat(series, axis=1).sort_index()
    return wide


# =========================
# Wind (fix duplicates!)
# =========================

def circular_mean_deg(arr_deg):
    arr = np.asarray(arr_deg, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    th = np.deg2rad(arr)
    s = np.nanmean(np.sin(th))
    c = np.nanmean(np.cos(th))
    ang = (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0
    return ang

def circular_smooth_deg(deg_series, window):
    theta = np.deg2rad(deg_series.to_numpy(dtype=float))
    sinv = pd.Series(np.sin(theta), index=deg_series.index)
    cosv = pd.Series(np.cos(theta), index=deg_series.index)
    sin_s = sinv.rolling(window, center=True, min_periods=max(2, window//2)).mean()
    cos_s = cosv.rolling(window, center=True, min_periods=max(2, window//2)).mean()
    out = (np.rad2deg(np.arctan2(sin_s, cos_s)) + 360.0) % 360.0
    return pd.Series(out, index=deg_series.index)

def read_wind_1min(t0, t1):
    df = pd.read_csv(WIND_CSV)

    # flexible column names
    cols = {c.strip(): c for c in df.columns}
    time_col = None
    for cand in ["DATE", "Date", "Datetime", "datetime", "time", "Time"]:
        if cand in cols:
            time_col = cols[cand]
            break
    if time_col is None:
        raise SystemExit(f"Wind CSV missing time column. Found: {list(df.columns)}")

    ws_col = None
    for cand in ["wind_speed", "Wind Speed", "windspeed", "WS", "ws"]:
        if cand in cols:
            ws_col = cols[cand]
            break
    if ws_col is None:
        raise SystemExit(f"Wind CSV missing wind_speed column. Found: {list(df.columns)}")

    wd_col = None
    for cand in ["wind_direction", "wind_dir", "Wind Direction", "wd", "WD"]:
        if cand in cols:
            wd_col = cols[cand]
            break
    if wd_col is None:
        raise SystemExit(f"Wind CSV missing wind_direction column. Found: {list(df.columns)}")

    df["dt_raw"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["dt_raw"]).copy()
    df["wind_speed"] = pd.to_numeric(df[ws_col], errors="coerce")
    df["wind_dir"]   = pd.to_numeric(df[wd_col], errors="coerce")

    # fix dir sentinels
    df.loc[(df["wind_dir"] >= 900) | (df["wind_dir"] < 0), "wind_dir"] = np.nan

    # timezone convert
    if WIND_TIME_IS_UTC:
        dt_utc = df["dt_raw"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
        dt_loc = dt_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        df["dt"] = dt_loc
    else:
        df["dt"] = df["dt_raw"]

    df = df.dropna(subset=["dt"]).copy()
    df = df.set_index("dt")[["wind_speed", "wind_dir"]].sort_index()

    # ✅ CRITICAL FIX: remove / aggregate duplicate timestamps BEFORE resample
    if df.index.duplicated().any():
        # speed mean, direction circular mean
        g = df.groupby(level=0)
        ws = g["wind_speed"].mean()
        wd = g["wind_dir"].apply(circular_mean_deg)
        df = pd.concat([ws, wd], axis=1)
        df.columns = ["wind_speed", "wind_dir"]
        df = df.sort_index()

    # buffer
    df = df.loc[(t0 - pd.Timedelta(hours=2)):(t1 + pd.Timedelta(hours=2))]

    # resample 1min
    df = df.resample("1min").ffill()

    # fill gaps
    df["wind_speed"] = df["wind_speed"].interpolate(limit=10).ffill().bfill()
    df["wind_dir"]   = df["wind_dir"].interpolate(limit=10).ffill().bfill()

    # smooth
    w = max(3, int(WIND_SMOOTH_MIN))
    df["wind_speed"] = df["wind_speed"].rolling(w, center=True, min_periods=max(2, w//2)).mean()
    df["wind_dir"]   = circular_smooth_deg(df["wind_dir"], window=w)

    return df.loc[t0:t1]


# =========================
# IDW
# =========================

def precompute_idw_weights(points_xy, sensor_xy, power=2.0, eps_m=5.0):
    P = points_xy.shape[0]
    Ns = sensor_xy.shape[0]
    W = np.zeros((P, Ns), dtype=float)
    for i in range(P):
        dx = sensor_xy[:, 0] - points_xy[i, 0]
        dy = sensor_xy[:, 1] - points_xy[i, 1]
        d = np.sqrt(dx*dx + dy*dy)
        d = np.maximum(d, eps_m)
        w = 1.0 / (d**power)
        s = np.sum(w)
        if s > 0:
            W[i, :] = w / s
    return W

def idw_field(W, sensor_vals):
    return W @ sensor_vals


# =========================
# Centered puff model (unit Q)
# =========================

def wind_from_to_unitvec(wd_from_deg):
    if not np.isfinite(wd_from_deg):
        return np.array([0.0, 1.0], dtype=float)
    wd_to = (wd_from_deg + 180.0) % 360.0
    th = np.deg2rad(wd_to)
    u = np.sin(th)
    v = np.cos(th)
    n = math.hypot(u, v)
    if n == 0:
        return np.array([0.0, 1.0], dtype=float)
    return np.array([u/n, v/n], dtype=float)

def predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy, sigma0_m, K_m2s, tau_s):
    """
    returns G: (T, Ns) for unit Q
    (center fixed at source; wind rotates+stretches)
    """
    T = len(times)
    Ns = sensor_xy.shape[0]
    G = np.zeros((T, Ns), dtype=float)
    release_idx = np.where(release_mask)[0].tolist()

    x = sensor_xy[:, 0]
    y = sensor_xy[:, 1]

    for i in range(T):
        out = np.zeros(Ns, dtype=float)
        for j in release_idx:
            if j > i:
                break
            age = (i - j) * DT_S
            age_eff = max(age, DT_S * 0.5)

            U = float(ws[j]) if np.isfinite(ws[j]) else 0.0
            U = min(max(U, 0.0), U_CAP)

            e_par = wind_from_to_unitvec(float(wd_from[j]) if np.isfinite(wd_from[j]) else np.nan)
            e_perp = np.array([-e_par[1], e_par[0]], dtype=float)

            sigma_perp = math.sqrt(max(1e-6, sigma0_m*sigma0_m + 2.0*(K_m2s*K_PERP_SCALE)*age_eff))
            stretch = 1.0 + (U * age_eff) / max(1.0, WIND_STRETCH_L)
            stretch = min(stretch, STRETCH_CAP)
            sigma_par = sigma_perp * stretch

            x_par  = x * e_par[0]  + y * e_par[1]
            x_perp = x * e_perp[0] + y * e_perp[1]

            inv = 1.0 / (2.0 * math.pi * sigma_par * sigma_perp)
            expo = np.exp(-0.5 * ((x_par/sigma_par)**2 + (x_perp/sigma_perp)**2))
            decay = math.exp(-age_eff / max(1.0, tau_s))

            out += (DT_S * inv) * expo * decay

        G[i, :] = out

    return G


# =========================
# Calibration (handle NaNs)
# =========================

def solve_Q_nonneg(G, y):
    mask = np.isfinite(y)
    if not np.any(mask):
        return 0.0
    g = G[mask]
    yy = y[mask]
    denom = float(np.dot(g, g))
    if denom <= 0:
        return 0.0
    Q = float(np.dot(g, yy) / denom)
    return max(0.0, Q)

def rmse_masked(pred, y):
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.nan
    d = (pred - y)[mask]
    return float(np.sqrt(np.mean(d*d)))

def grid_search_calibration(y_obs, times, ws, wd_from, release_mask, sensor_xy):
    sigma0_list = [120, 180, 240, 300]
    K_list      = [120, 250, 400, 650]
    tau_list    = [600, 900, 1500, 2400, 3600]

    best = None
    best_rmse = 1e99

    for sigma0 in sigma0_list:
        for K in K_list:
            for tau in tau_list:
                G = predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy,
                                             sigma0_m=sigma0, K_m2s=K, tau_s=tau)
                Q = solve_Q_nonneg(G, y_obs)
                pred = Q * G
                e = rmse_masked(pred, y_obs)
                if np.isfinite(e) and e < best_rmse:
                    best_rmse = e
                    best = {"sigma0": sigma0, "K": K, "tau": tau, "Q": Q, "rmse": e}

    return best


# =========================
# Main
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    t1 = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")
    fw0 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_START}")
    fw1 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_END}")
    t_baseline_end = pd.Timestamp(f"{EVENT_DATE} {BASELINE_END}")
    t_fit_end      = pd.Timestamp(f"{EVENT_DATE} {FIT_END}")

    # 1) sensors
    wide = load_all_sensors_1min(t0, t1)
    sensors = [s for s in ["S1","S2","S3","S4","S5","S6"] if s in wide.columns]
    wide = wide[sensors]

    # 2) wind (fixed duplicates)
    wind = read_wind_1min(t0, t1)

    # 3) align
    idx = wide.index.intersection(wind.index)
    wide = wide.loc[idx]
    wind = wind.loc[idx]

    # 4) fit window
    fit_idx = wide.index[(wide.index >= t0) & (wide.index <= t_fit_end)]
    wide_fit = wide.loc[fit_idx]
    wind_fit = wind.loc[fit_idx]

    # 5) baseline (median)
    base_idx = wide.index[(wide.index >= t0) & (wide.index <= t_baseline_end)]
    baselines = wide.loc[base_idx].median(axis=0, skipna=True)

    y = (wide_fit - baselines).clip(lower=0.0)  # enhancement
    # keep NaNs (masked calibration)
    y_arr = y.to_numpy(dtype=float)

    # 6) sensor xy
    sensor_xy = np.array([latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], SRC_LAT, SRC_LON) for s in sensors], dtype=float)

    # 7) wind arrays
    ws_fit = wind_fit["wind_speed"].to_numpy(dtype=float)
    wd_fit = wind_fit["wind_dir"].to_numpy(dtype=float)

    times_fit = wide_fit.index.to_pydatetime()

    # 8) release mask (fit)
    release_fit = np.array([(t >= fw0) and (t < fw1) for t in wide_fit.index], dtype=bool)

    # 9) calibrate
    best = grid_search_calibration(y_arr, times_fit, ws_fit, wd_fit, release_fit, sensor_xy)
    print("\n=== BEST PARAMS (grid search) ===")
    print(best)

    # 10) timeseries check plot
    G_best = predict_unitQ_at_sensors(times_fit, ws_fit, wd_fit, release_fit, sensor_xy,
                                     sigma0_m=best["sigma0"], K_m2s=best["K"], tau_s=best["tau"])
    pred_enh = best["Q"] * G_best
    pred = pred_enh + baselines.to_numpy(dtype=float)

    pred_df = pd.DataFrame(pred, index=wide_fit.index, columns=sensors)
    obs_df  = wide_fit.copy()

    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.2*len(sensors)), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    for ax, s in zip(axes, sensors):
        ax.plot(obs_df.index, obs_df[s].values, label=f"{s} observed", linewidth=1.0)
        ax.plot(pred_df.index, pred_df[s].values, label=f"{s} model", linewidth=1.0)
        ax.axvspan(fw0, fw1, color="gray", alpha=0.15)
        ax.set_ylabel("PM2.5 (ug/m3)")
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time (local)")
    fig.suptitle("Observed vs Calibrated Centered Gaussian Puff Model")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "calibration_timeseries.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved:", out_png)

    # 11) full times
    full_times = pd.date_range(t0, t1, freq="1min")
    wide_full = wide.reindex(full_times).interpolate(limit=3).ffill().bfill()
    wind_full = wind.reindex(full_times).interpolate(limit=10).ffill().bfill()

    ws_full = wind_full["wind_speed"].to_numpy(dtype=float)
    wd_full = wind_full["wind_dir"].to_numpy(dtype=float)

    release_full = np.array([(t >= fw0) and (t < fw1) for t in full_times], dtype=bool)

    # 12) grid points
    # ===== NEW: set grid radius by sensor spread (no square look) =====
    sensor_r = np.sqrt(sensor_xy[:,0]**2 + sensor_xy[:,1]**2)
    R = float(np.nanmax(sensor_r)) * 1.25   # 25% buffer, you can use 1.4 if you want more

    xs = np.linspace(-R, R, GRID_N)
    ys = np.linspace(-R, R, GRID_N)
    X, Y = np.meshgrid(xs, ys)

    points_xy = np.column_stack([X.ravel(), Y.ravel()])

    # Keep only points inside a circle (removes square boundary)
    mask = (points_xy[:,0]**2 + points_xy[:,1]**2) <= (R**2)
    points_xy = points_xy[mask]

    lat_grid, lon_grid = xy_to_latlon_grid(X, Y, SRC_LAT, SRC_LON)
    lat_points = lat_grid.ravel()[mask]
    lon_points = lon_grid.ravel()[mask]

    # 13) IDW weights
    W_idw = precompute_idw_weights(points_xy, sensor_xy, power=2.0, eps_m=5.0)

    # 14) model enhancement at sensors for ALL times (to separate background after fireworks)
    times_full = full_times.to_pydatetime()
    G_full_sensors = predict_unitQ_at_sensors(times_full, ws_full, wd_full, release_full, sensor_xy,
                                             sigma0_m=best["sigma0"], K_m2s=best["K"], tau_s=best["tau"])
    enh_sensors = best["Q"] * G_full_sensors  # (T,Ns)

    obs_sensors = wide_full[sensors].to_numpy(dtype=float)  # (T,Ns)
    bg_sensors = obs_sensors - enh_sensors
    bg_sensors = np.clip(bg_sensors, 0.0, None)

    # 15) enhancement grid per frame (centered)
    release_idx = np.where(release_full)[0].tolist()

    enh_grids = []
    total_grids = []

    for i in range(len(full_times)):
        # background from (obs - model_enh) at sensors
        bg_vals = bg_sensors[i, :]
        bg_grid = idw_field(W_idw, bg_vals)

        # enhancement on grid
        enh = np.zeros(points_xy.shape[0], dtype=float)
        for j in release_idx:
            if j > i:
                break
            age = (i - j) * DT_S
            age_eff = max(age, DT_S * 0.5)

            U = float(ws_full[j]) if np.isfinite(ws_full[j]) else 0.0
            U = min(max(U, 0.0), U_CAP)

            e_par = wind_from_to_unitvec(float(wd_full[j]) if np.isfinite(wd_full[j]) else np.nan)
            e_perp = np.array([-e_par[1], e_par[0]], dtype=float)

            sigma_perp = math.sqrt(max(1e-6, best["sigma0"]**2 + 2.0*(best["K"]*K_PERP_SCALE)*age_eff))
            stretch = 1.0 + (U * age_eff) / max(1.0, WIND_STRETCH_L)
            stretch = min(stretch, STRETCH_CAP)
            sigma_par = sigma_perp * stretch

            x = points_xy[:, 0]
            yv = points_xy[:, 1]
            x_par  = x * e_par[0]  + yv * e_par[1]
            x_perp = x * e_perp[0] + yv * e_perp[1]

            inv = 1.0 / (2.0 * math.pi * sigma_par * sigma_perp)
            expo = np.exp(-0.5 * ((x_par/sigma_par)**2 + (x_perp/sigma_perp)**2))
            decay = math.exp(-age_eff / max(1.0, best["tau"]))
            enh += (DT_S * inv) * expo * decay

        enh *= float(best["Q"])
        total = bg_grid + enh

        enh_grids.append(enh)
        total_grids.append(total)

    # 16) choose CMAX
    all_vals = np.concatenate([np.ravel(tg) for tg in total_grids]).astype(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise SystemExit("No finite grid values. Check your inputs.")

    if CMAX_AUTO:
        cmax = float(np.nanpercentile(all_vals, 97))
        cmax = max(60.0, min(cmax, 350.0))
    else:
        cmax = float(CMAX_FIXED)

    print("DEBUG: CMAX used =", cmax)

    # 17) frames
    frames = []
    labels = []

    for i, t in enumerate(full_times):
        total = np.clip(total_grids[i], 0.0, cmax)
        w = (total / cmax) ** GAMMA
        w = np.clip(w, 0.0, 1.0)

        good = np.where(w >= WMIN)[0]

        # if too many points, keep top
        if good.size > MAX_POINTS_PER_FRAME:
            top = np.argsort(w[good])[-MAX_POINTS_PER_FRAME:]
            good = good[top]

        frame = [[float(lat_points[k]), float(lon_points[k]), float(w[k])] for k in good]
        frames.append(frame)
        labels.append(t.strftime("%H:%M"))

    print("DEBUG: frames =", len(frames))
    print("DEBUG: points in first frame =", len(frames[0]) if frames else None)
    print("DEBUG: total points =", sum(len(fr) for fr in frames))
    if sum(len(fr) for fr in frames) == 0:
        print("WARNING: all frames empty. Try CMAX_AUTO=False and CMAX_FIXED=120, or set WMIN=0.0 (already).")

    # 18) map
    m = folium.Map(location=(SRC_LAT, SRC_LON), zoom_start=13, tiles="OpenStreetMap")
    folium.Marker((SRC_LAT, SRC_LON), popup="Fireworks Source (fixed center)").add_to(m)
    for s in sensors:
        slat, slon = SENSOR_LATLON[s]
        folium.CircleMarker((slat, slon), radius=6, popup=s, fill=True).add_to(m)

    # Heat layer
    try:
        HeatMapWithTime(
            frames,
            index=labels,
            radius=HEAT_RADIUS,
            min_opacity=MIN_OPACITY,
            max_opacity=MAX_OPACITY,
            use_local_extrema=False,
            auto_play=False,
            display_index=True
        ).add_to(m)
    except TypeError:
        HeatMapWithTime(
            frames,
            index=labels,
            radius=HEAT_RADIUS,
            min_opacity=MIN_OPACITY,
            use_local_extrema=False,
            auto_play=False
        ).add_to(m)

    out_html = os.path.join(OUT_DIR, "calibrated_centered_puff_heatmap_time.html")
    m.save(out_html)
    print("\nSaved:", out_html)
    print("\nHow to view:")
    print("  cd", OUT_DIR)
    print("  python -m http.server 8000")
    print("  http://localhost:8000/calibrated_centered_puff_heatmap_time.html")


if __name__ == "__main__":
    main()
