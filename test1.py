#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fireworks_centered_puff_calibrate_heatmap.py

Centered Gaussian Puff (source fixed) + wind-shape (rotate+stretch) + calibration with sensors
- Emission only during fireworks window
- After fireworks: diffusion + exponential decay (tau), no new emission
- Background field inferred from sensors via (observed - modeled_enhancement) and IDW
- S4 robust linear calibration to reference (median of other sensors) using pre-fireworks window
- Robust MAD clipping each minute to prevent single-sensor dominating the map
- Continuous optimization for (sigma0, K, tau) using SciPy minimize if available
  (Q is always solved analytically; NEVER grid-searched)

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

# SciPy optional
try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =========================
# USER SETTINGS (edit here)
# =========================

EVENT_DATE = "2025-07-04"
START_TIME = "20:00:00"
END_TIME   = "23:30:00"

FIREWORKS_START = "22:20:00"
FIREWORKS_END   = "22:40:00"

# calibration windows
BASELINE_END = "22:10:00"   # baseline / S4-cal window end (pre-fireworks)
FIT_END      = "23:00:00"   # fit window end

# inputs
SENSOR_GLOB = "Data_*.csv"
WIND_CSV    = "wind_speed_2025.csv"

# Fireworks source (center fixed, no drifting)
SRC_LAT, SRC_LON = 47.6403, -122.3352

# Sensor locations (edit S4 if needed)
SENSOR_LATLON = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    # "S4": (47.63629, -122.33998)
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

PM_COL_CANDIDATES = ["PM2.5_Env", "PM2.5_Std", "PM2.5"]

# wind time handling
WIND_TIME_IS_UTC = True
LOCAL_TZ = "America/Los_Angeles"

# grid geometry
GRID_N = 35
USE_CIRCULAR_GRID_MASK = True   # removes square boundary
GRID_BUFFER_FACTOR = 1.35       # based on max sensor radius

# heatmap style
HEAT_RADIUS = 22
MIN_OPACITY = 0.25
MAX_OPACITY = 0.95

# normalization
CMAX_AUTO = True
CMAX_FIXED = 200.0
GAMMA = 0.65
WMIN = 0.0
MAX_POINTS_PER_FRAME = 2500

# physics knobs (USED)
DT_S = 60.0

WIND_STRETCH_L = 1500.0   # bigger => less extreme wind sudden changes
STRETCH_CAP    = 6.0
U_CAP          = 6.0
K_PERP_SCALE   = 1.0

# wind smoothing
WIND_SMOOTH_MIN = 7

# S4 / background robustness
MAD_CLIP_K = 3.5

# output
OUT_DIR = "calibrated_centered_outputs"


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

def xy_to_latlon_points(xy, lat0, lon0):
    R = 6371000.0
    lat0_r = np.deg2rad(lat0)
    lon0_r = np.deg2rad(lon0)
    X = xy[:, 0]
    Y = xy[:, 1]
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
        if snum == 4:
            continue
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
# Robust stats (S4 + clipping)
# =========================

def robust_linear_calibrate_to_reference(x, y, k=1.5, iters=15):
    """
    Fit y ≈ a*x + b using robust IRLS (Huber-like).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    eps = 1e-6
    for _ in range(iters):
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + eps
        u = r / (k * s)
        w = np.ones_like(u)
        mask = np.abs(u) > 1.0
        w[mask] = 1.0 / np.abs(u[mask])
        Aw = A * w[:, None]
        yw = y * w
        a, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    return float(a), float(b)

def calibrate_S4(wide, t_start, t_end, target="S4"):
    """
    Calibrate S4 to match reference=median(other sensors) on pre-fireworks window.
    """
    if target not in wide.columns:
        return wide
    cols = [c for c in wide.columns if c != target]
    if len(cols) < 2:
        return wide

    sub = wide.loc[(wide.index >= t_start) & (wide.index <= t_end), [target] + cols].copy()
    ref = sub[cols].median(axis=1, skipna=True)

    x = sub[target].to_numpy(dtype=float)
    y = ref.to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return wide

    a, b = robust_linear_calibrate_to_reference(x[m], y[m])
    wide2 = wide.copy()
    wide2[target] = a * wide2[target] + b
    print(f"[S4 CAL] applied: {target}_cal = {a:.4g}*{target} + {b:.4g}")
    return wide2

def clip_by_mad(vals, k=MAD_CLIP_K):
    v = np.asarray(vals, float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad < 1e-6:
        return v
    lo = med - k * mad
    hi = med + k * mad
    return np.clip(v, lo, hi)


# =========================
# Wind (fix duplicates + circular ops)
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
    sin_s = sinv.rolling(window, center=True, min_periods=max(2, window // 2)).mean()
    cos_s = cosv.rolling(window, center=True, min_periods=max(2, window // 2)).mean()
    out = (np.rad2deg(np.arctan2(sin_s, cos_s)) + 360.0) % 360.0
    return pd.Series(out, index=deg_series.index)

def read_wind_1min(t0, t1):
    df = pd.read_csv(WIND_CSV)
    cols = {c.strip(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    time_col = pick(["DATE", "Date", "Datetime", "datetime", "time", "Time"])
    ws_col   = pick(["wind_speed", "Wind Speed", "windspeed", "WS", "ws"])
    wd_col   = pick(["wind_direction", "wind_dir", "Wind Direction", "WD", "wd"])

    if time_col is None or ws_col is None or wd_col is None:
        raise SystemExit(f"Wind CSV columns not recognized. Found: {list(df.columns)}")

    df["dt_raw"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["dt_raw"]).copy()

    df["wind_speed"] = pd.to_numeric(df[ws_col], errors="coerce")
    df["wind_dir"]   = pd.to_numeric(df[wd_col], errors="coerce")

    df.loc[(df["wind_dir"] >= 900) | (df["wind_dir"] < 0), "wind_dir"] = np.nan

    if WIND_TIME_IS_UTC:
        dt_utc = df["dt_raw"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
        dt_loc = dt_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        df["dt"] = dt_loc
    else:
        df["dt"] = df["dt_raw"]

    df = df.dropna(subset=["dt"]).copy()
    df = df.set_index("dt")[["wind_speed", "wind_dir"]].sort_index()

    # fix duplicate timestamps BEFORE resample
    if df.index.duplicated().any():
        g = df.groupby(level=0)
        ws = g["wind_speed"].mean()
        wd = g["wind_dir"].apply(circular_mean_deg)
        df = pd.concat([ws, wd], axis=1)
        df.columns = ["wind_speed", "wind_dir"]
        df = df.sort_index()

    df = df.loc[(t0 - pd.Timedelta(hours=2)):(t1 + pd.Timedelta(hours=2))]
    df = df.resample("1min").ffill()
    df["wind_speed"] = df["wind_speed"].interpolate(limit=10).ffill().bfill()
    df["wind_dir"]   = df["wind_dir"].interpolate(limit=10).ffill().bfill()

    w = max(3, int(WIND_SMOOTH_MIN))
    df["wind_speed"] = df["wind_speed"].rolling(w, center=True, min_periods=max(2, w // 2)).mean()
    df["wind_dir"]   = circular_smooth_deg(df["wind_dir"], window=w)

    return df.loc[t0:t1]


# =========================
# IDW weights
# =========================

def precompute_idw_weights(points_xy, sensor_xy, power=2.0, eps_m=5.0):
    P = points_xy.shape[0]
    Ns = sensor_xy.shape[0]
    W = np.zeros((P, Ns), dtype=float)
    for i in range(P):
        dx = sensor_xy[:, 0] - points_xy[i, 0]
        dy = sensor_xy[:, 1] - points_xy[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, eps_m)
        w = 1.0 / (d ** power)
        s = np.sum(w)
        if s > 0:
            W[i, :] = w / s
    return W

def idw_field(W, sensor_vals):
    return W @ sensor_vals


# =========================
# Centered puff model (unit Q at sensors)
# =========================

def wind_from_to_unitvec(wd_from_deg):
    """
    Convert meteorological FROM direction to unit vector pointing TO direction.
    Returns unit vector in (east, north).
    """
    if not np.isfinite(wd_from_deg):
        return np.array([0.0, 1.0], dtype=float)
    wd_to = (wd_from_deg + 180.0) % 360.0
    th = np.deg2rad(wd_to)
    u = np.sin(th)
    v = np.cos(th)
    n = math.hypot(u, v)
    if n == 0:
        return np.array([0.0, 1.0], dtype=float)
    return np.array([u / n, v / n], dtype=float)

def predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy,
                            sigma0_m, K_m2s, tau_s):
    """
    G(t,s): concentration at sensor s at time t for unit emission rate Q.
    Center is fixed at source. Wind affects ellipse orientation + stretch, but NOT center.
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

            sigma_perp = math.sqrt(max(1e-6, sigma0_m * sigma0_m + 2.0 * (K_m2s * K_PERP_SCALE) * age_eff))
            stretch = 1.0 + (U * age_eff) / max(1.0, WIND_STRETCH_L)
            stretch = min(stretch, STRETCH_CAP)
            sigma_par = sigma_perp * stretch

            x_par  = x * e_par[0]  + y * e_par[1]
            x_perp = x * e_perp[0] + y * e_perp[1]

            inv = 1.0 / (2.0 * math.pi * sigma_par * sigma_perp)
            expo = np.exp(-0.5 * ((x_par / sigma_par) ** 2 + (x_perp / sigma_perp) ** 2))
            decay = math.exp(-age_eff / max(1.0, tau_s))

            out += (DT_S * inv) * expo * decay

        G[i, :] = out

    return G


# =========================
# Fit Q analytically + RMSE
# =========================

def fit_Q_and_rmse(y, G, w=None):
    """
    y, G: (T, Ns) arrays. Mask NaNs automatically.
    Q* = sum(w*y*G)/sum(w*G^2) with nonneg constraint.
    """
    yv = np.asarray(y, float)
    Gv = np.asarray(G, float)
    if w is None:
        wv = np.ones_like(yv)
    else:
        wv = np.asarray(w, float)

    mask = np.isfinite(yv) & np.isfinite(Gv) & np.isfinite(wv)
    if not np.any(mask):
        return 0.0, np.nan

    yg = (wv * yv * Gv)[mask]
    gg = (wv * Gv * Gv)[mask]
    denom = float(np.sum(gg))
    if denom <= 0:
        return 0.0, np.nan

    Q = float(np.sum(yg) / denom)
    Q = max(0.0, Q)

    pred = Q * Gv
    diff = (pred - yv)[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return Q, rmse


# =========================
# Dense grid fallback (if no SciPy)
# =========================

def grid_search_dense(y, times, ws, wd_from, release_mask, sensor_xy, w=None):
    sigma0_list = np.linspace(60, 520, 12)     # denser than before
    K_list      = np.geomspace(20, 1500, 12)
    tau_list    = np.geomspace(200, 12000, 12)

    best = {"rmse": 1e99}
    for sigma0 in sigma0_list:
        for K in K_list:
            for tau in tau_list:
                G = predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy,
                                             sigma0_m=float(sigma0), K_m2s=float(K), tau_s=float(tau))
                Q, rmse = fit_Q_and_rmse(y, G, w=w)
                if np.isfinite(rmse) and rmse < best["rmse"]:
                    best = {"sigma0": float(sigma0), "K": float(K), "tau": float(tau), "Q": float(Q), "rmse": float(rmse)}
    return best


# =========================
# Continuous optimization (SciPy)
# =========================

def fit_params_continuous(y, times, ws, wd_from, release_mask, sensor_xy, w=None, n_starts=12):
    """
    Continuous optimization over (sigma0, K, tau); Q solved analytically each evaluation.
    Use multi-start. Increase n_starts on lab PC (e.g., 30~50).
    """
    if w is None:
        w = np.ones_like(y)

    # log-bounds
    log_bounds = [
        (np.log(20.0),   np.log(900.0)),     # sigma0 (m)
        (np.log(5.0),    np.log(2500.0)),    # K (m^2/s)
        (np.log(120.0),  np.log(30000.0)),   # tau (s)
    ]

    def objective(p_log):
        sigma0 = float(np.exp(p_log[0]))
        K      = float(np.exp(p_log[1]))
        tau    = float(np.exp(p_log[2]))

        G = predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy,
                                     sigma0_m=sigma0, K_m2s=K, tau_s=tau)
        _, rmse = fit_Q_and_rmse(y, G, w=w)
        return float(rmse)

    # start near a dense-grid best (for stability)
    coarse = grid_search_dense(y, times, ws, wd_from, release_mask, sensor_xy, w=w)
    p0 = np.log([coarse["sigma0"], coarse["K"], coarse["tau"]])

    best = {"rmse": 1e99}
    rng = np.random.default_rng(0)

    starts = [p0]
    for _ in range(max(0, int(n_starts) - 1)):
        s = rng.uniform([b[0] for b in log_bounds], [b[1] for b in log_bounds])
        starts.append(s)

    for p_init in starts:
        res = minimize(objective, p_init, method="L-BFGS-B", bounds=log_bounds, options={"maxiter": 220})
        if not res.success:
            continue
        sigma0, K, tau = np.exp(res.x)

        G = predict_unitQ_at_sensors(times, ws, wd_from, release_mask, sensor_xy,
                                     sigma0_m=float(sigma0), K_m2s=float(K), tau_s=float(tau))
        Q, rmse = fit_Q_and_rmse(y, G, w=w)
        if np.isfinite(rmse) and rmse < best["rmse"]:
            best = {"sigma0": float(sigma0), "K": float(K), "tau": float(tau), "Q": float(Q), "rmse": float(rmse)}

    # fallback
    if best["rmse"] >= 1e98:
        return coarse
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

    # 1) sensors 1-min
    wide = load_all_sensors_1min(t0, t1)
    sensors = [s for s in ["S1","S2","S3","S4","S5","S6"] if s in wide.columns]
    wide = wide[sensors]

    # 2) wind 1-min
    wind = read_wind_1min(t0, t1)

    # 3) align
    idx = wide.index.intersection(wind.index)
    wide = wide.loc[idx]
    wind = wind.loc[idx]

    # 4) S4 calibration (pre-fireworks)
    calib_end = min(t_baseline_end, fw0 - pd.Timedelta(minutes=5))
    wide = calibrate_S4(wide, t0, calib_end, target="S4")

    # 5) fit window
    fit_idx = wide.index[(wide.index >= t0) & (wide.index <= t_fit_end)]
    wide_fit = wide.loc[fit_idx]
    wind_fit = wind.loc[fit_idx]

    # 6) baseline (median pre-fireworks)
    base_idx = wide.index[(wide.index >= t0) & (wide.index <= t_baseline_end)]
    baselines = wide.loc[base_idx].median(axis=0, skipna=True)

    # enhancement y (keep NaNs; clip negative to 0)
    y = (wide_fit - baselines).clip(lower=0.0)
    y_arr = y.to_numpy(dtype=float)

    # 7) sensor_xy relative to source (0,0)
    sensor_xy = np.array([latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], SRC_LAT, SRC_LON) for s in sensors], dtype=float)

    # 8) wind arrays for fitting
    ws_fit = wind_fit["wind_speed"].to_numpy(dtype=float)
    wd_fit = wind_fit["wind_dir"].to_numpy(dtype=float)
    times_fit = wide_fit.index.to_pydatetime()

    # 9) release mask (fit)
    release_fit = np.array([(t >= fw0) and (t < fw1) for t in wide_fit.index], dtype=bool)

    # 10) fit parameters
    if HAVE_SCIPY:
        # increase n_starts on lab PC (e.g. 30~50)
        best = fit_params_continuous(y_arr, times_fit, ws_fit, wd_fit, release_fit, sensor_xy, n_starts=14)
    else:
        best = grid_search_dense(y_arr, times_fit, ws_fit, wd_fit, release_fit, sensor_xy)

    print("\n=== BEST PARAMS (continuous fit if SciPy available) ===")
    print(best)

    # 11) diagnostic timeseries plot (fit window)
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

    # 12) full timeline (8pm..end)
    full_times = pd.date_range(t0, t1, freq="1min")
    wide_full = wide.reindex(full_times).interpolate(limit=3).ffill().bfill()
    wind_full = wind.reindex(full_times).interpolate(limit=10).ffill().bfill()

    ws_full = wind_full["wind_speed"].to_numpy(dtype=float)
    wd_full = wind_full["wind_dir"].to_numpy(dtype=float)
    times_full = full_times.to_pydatetime()

    release_full = np.array([(t >= fw0) and (t < fw1) for t in full_times], dtype=bool)

    # 13) compute modeled enhancement at sensors for full times
    G_full_sensors = predict_unitQ_at_sensors(times_full, ws_full, wd_full, release_full, sensor_xy,
                                             sigma0_m=best["sigma0"], K_m2s=best["K"], tau_s=best["tau"])
    enh_sensors = best["Q"] * G_full_sensors

    obs_sensors = wide_full[sensors].to_numpy(dtype=float)
    bg_sensors = obs_sensors - enh_sensors
    bg_sensors = np.clip(bg_sensors, 0.0, None)

    # 14) build grid points (mask circle if desired)
    sensor_r = np.sqrt(sensor_xy[:, 0]**2 + sensor_xy[:, 1]**2)
    R = float(np.nanmax(sensor_r)) * float(GRID_BUFFER_FACTOR)
    xs = np.linspace(-R, R, GRID_N)
    ys = np.linspace(-R, R, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    points_xy = np.column_stack([X.ravel(), Y.ravel()])

    if USE_CIRCULAR_GRID_MASK:
        mask = (points_xy[:, 0]**2 + points_xy[:, 1]**2) <= (R**2)
        points_xy = points_xy[mask]
    else:
        mask = None

    lat_points, lon_points = xy_to_latlon_points(points_xy, SRC_LAT, SRC_LON)

    # 15) IDW weights
    W_idw = precompute_idw_weights(points_xy, sensor_xy, power=2.0, eps_m=5.0)

    # 16) compute enhancement on grid for each time (center fixed)
    release_idx = np.where(release_full)[0].tolist()

    total_grids = []
    for i in range(len(full_times)):
        # background grid (robust clipped)
        bg_vals = bg_sensors[i, :].copy()
        bg_vals = clip_by_mad(bg_vals, k=MAD_CLIP_K)
        bg_grid = idw_field(W_idw, bg_vals)

        # enhancement grid
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
        total_grids.append(total)

    # 17) CMAX
    all_vals = np.concatenate([np.ravel(tg) for tg in total_grids]).astype(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise SystemExit("No finite grid values.")

    if CMAX_AUTO:
        cmax = float(np.nanpercentile(all_vals, 97))
        cmax = max(60.0, min(cmax, 350.0))
    else:
        cmax = float(CMAX_FIXED)

    print("DEBUG: CMAX used =", cmax)

    # 18) frames for HeatMapWithTime
    frames = []
    labels = []
    for i, t in enumerate(full_times):
        total = np.clip(total_grids[i], 0.0, cmax)
        wgt = (total / cmax) ** float(GAMMA)
        wgt = np.clip(wgt, 0.0, 1.0)

        good = np.where(wgt >= WMIN)[0]
        if good.size > MAX_POINTS_PER_FRAME:
            top = np.argsort(wgt[good])[-MAX_POINTS_PER_FRAME:]
            good = good[top]

        frame = [[float(lat_points[k]), float(lon_points[k]), float(wgt[k])] for k in good]
        frames.append(frame)
        labels.append(t.strftime("%H:%M"))

    print("DEBUG: frames =", len(frames))
    print("DEBUG: points in first frame =", len(frames[0]) if frames else None)
    print("DEBUG: total points =", sum(len(fr) for fr in frames))

    # 19) map
    m = folium.Map(location=(SRC_LAT, SRC_LON), zoom_start=13, tiles="OpenStreetMap")
    folium.Marker((SRC_LAT, SRC_LON), popup="Fireworks Source (fixed center)").add_to(m)
    for s in sensors:
        slat, slon = SENSOR_LATLON[s]
        folium.CircleMarker((slat, slon), radius=6, popup=s, fill=True).add_to(m)

    # heat layer
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
