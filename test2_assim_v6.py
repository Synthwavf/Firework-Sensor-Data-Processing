#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test2_assim_v6.py

Baseline-only inversion + centered diffusion (no drift) + wind-informed anisotropy.

Project-level requirements summary (data -> baseline -> physics -> inversion -> viz -> outputs):

1) Goal
- Use multi-sensor PM2.5 + wind to generate a time-varying plume heatmap for paper-grade visuals.
- Animate the time series on a map.

2) Inputs
- Sensor CSV per sensor: Data_*.csv with Counter and PM2.5_* columns.
- Wind CSV: wind_speed_2025.csv with DATE, wind_speed, wind_direction ("from"; 999/invalid handled).
- Fixed sensor lat/lon and fixed source (fireworks location).

3) Baseline
- Use median of 22:15-22:20 per sensor (not 20:00).
- Baseline is fixed; baseline floor is transparent.
- Enhancement only: max(0, PM - baseline).

4) Physics / diffusion
- Center-locked plume (no advection shift).
- Wind shapes anisotropy only; avoid drifting the center.
- Auto domain sizing (AUTO_DOMAIN).

5) Inversion
- Use all sensors + wind + enhancement to invert Q(t).
- Avoid empty-looking fields; ensure sensors drive results (not wind alone).
- Provide diagnostics (sensor fit, emission profile).

6) Decay / end behavior
- Enhancement below baseline is transparent; plume fades out if no enhancement.
- Optional extra fade after fireworks end.

7) Visualization
- Plotly time heatmap (primary).
- Folium HeatMapWithTime (optional), with slider/blur compatibility handling.
- Baseline floor is transparent; sensor overlay enabled.

8) Outputs (OUT_DIR):
- baseline_values.csv
- sensor_weights.csv
- emission_profile.csv + emission_profile.png
- field_max_timeseries.png
- sensor_fit_enhancement.png
- centered_baseline_plume_plotly.html  (recommended)
- centered_baseline_plume_folium.html  (optional)

9) Acceptance checks
- Heatmap renders and animates (not blank).
- Diffusion is stepwise and center-locked to the source.
- Baseline uses 22:15-22:20 median; baseline floor is transparent.
- Wind shapes the plume but does not fully dominate it.
- Sensor fit diagnostics show sensor contributions are used.

This script models ONLY enhancement above baseline:
  enh_s(t) = max(0, PM_s(t) - baseline_s)

It then inverts Q(t) during the fireworks window [FIREWORKS_START, FIREWORKS_END),
and simulates a center-locked puff field:
  E(x,y,t) = sum_j Q_j * kernel_centered(x,y, age=t-tj; wind, sigma0, K, tau)

Rendering is enhancement only (baseline is implicitly 0),
and values below a display threshold are made transparent.

Run:
  python test2_assim_v6.py

View:
  python -m http.server 8000
  http://localhost:8000/<OUT_DIR>/centered_baseline_plume_plotly.html
"""

from __future__ import annotations

import os
import re
import glob
import math
import warnings
from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False

try:
    import folium
    from folium.plugins import HeatMapWithTime
    _HAVE_FOLIUM = True
except Exception:
    _HAVE_FOLIUM = False

from scipy.optimize import lsq_linear

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG (edit these)
# =========================
EVENT_DATE = "2025-07-04"

DEVICE_ON_TIME = "20:00"     # Counter=0 anchor time
ANALYSIS_START = "22:15"
END_TIME       = "23:30"

# Baseline window (YOUR requirement)
BASELINE_START = "22:15"
BASELINE_END   = "22:20"

# Emission inversion window (fireworks)
FIREWORKS_START = "22:20"
FIREWORKS_END   = "22:40"

SENSOR_GLOB = "Data_*.csv"
WIND_CSV    = "wind_speed_2025.csv"
OUT_DIR     = "paper_grade_outputs_v4b"

# Source (center lock)
SRC_LAT = 47.6403
SRC_LON = -122.3352

# Sensor lat/lon
SENSOR_LATLON: Dict[str, Tuple[float, float]] = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    "S4": (47.63629, -122.33998),
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

DT_MIN = 1  # time step minutes

# Domain/grid
AUTO_DOMAIN = True
GRID_N = 151
MIN_HALF_WIDTH_M = 2500.0
MAX_HALF_WIDTH_M = 12000.0
TARGET_POINTS_PER_FRAME = 1100

# Puff params search grid
SIGMA0_GRID = [120, 180, 250, 350]               # meters
K_GRID      = [50, 120, 250, 400, 650]           # m^2/s
TAU_GRID    = [300, 600, 900, 1200, 1800]        # seconds (tail decay)

# Wind shaping (NO advection shift; only anisotropy orientation/ratio)
PLUME_MODE  = "mild_wind"     # "isotropic" or "mild_wind"
STRETCH_CAP = 1.4             # lower => less wind-dominated shape

# Residual assimilation overlay (optional; keeps center-lock plume but locally nudges field toward sensors)
ENABLE_ASSIMILATION = True
ASSIM_ALPHA = 0.60          # 0 disables effect even if enabled
ASSIM_SIGMA_M = 250.0       # Gaussian sigma (meters). Smaller => more local sensor "nudge"
ASSIM_CLIP = 25.0           # clip correction at each grid point to +/- this (µg/m³)
ASSIM_MIN_WEIGHT = 1e-6     # if sum of Gaussian weights below this => no correction

# Sensor imprint (forces some color near sensors based on observed enhancement; helps when grid is sparse)
ENABLE_SENSOR_IMPRINT = True
IMPRINT_ALPHA = 0.85          # strength of imprint (0 disables effect even if enabled)
IMPRINT_SIGMA_M = 180.0       # meters; smaller => tighter sensor glow
IMPRINT_CLIP = 60.0           # clip imprint contribution at each grid point (µg/m³)
IMPRINT_MIN_ENH = 0.05        # ignore sensors with obs_enh below this

# Folium-only: add extra jittered points around each sensor to avoid 'blank around sensor' when grid is sparse
ADD_SENSOR_CLOUD = True
SENSOR_CLOUD_N = 24
SENSOR_CLOUD_RADIUS_M = 180.0
SENSOR_CLOUD_SCALE = 1.00
SENSOR_CLOUD_SEED = 7

# Extra post-fireworks fade (stronger/clearer decay than kernel tau alone). 0 disables.
POST_EVENT_DECAY_TAU = 900.0  # seconds; field *= exp(-(t - FIREWORKS_END)/POST_EVENT_DECAY_TAU) for t>FIREWORKS_END


# Robust sensor weights
FORCE_EXCLUDE_SENSORS: List[str] = []   # e.g. ["S4"] if it dominates
WEIGHT_CAP_MIN = 0.2
WEIGHT_CAP_MAX = 5.0

# Inversion smoothness (bigger => smoother Q)
LAMBDA_SMOOTH = 0.8

# IDW (for optional folium baseline floor; not required for plume-only)
IDW_POWER = 2.0
IDW_EPS_M = 5.0

# Visualization thresholds
AUTO_CMAX = True
DEFAULT_CMAX = 25.0

# Make < Z_MIN_DISPLAY transparent (display-only)
AUTO_ZMIN = True
DEFAULT_ZMIN = 0.25   # µg/m³

SHOW_SENSOR_OVERLAY = True

# Folium params (important: blur <= 1 to avoid your browser error)
FOLIUM_RADIUS = 22
FOLIUM_BLUR = 0.85
FOLIUM_WEIGHT_MIN_KEEP = 1e-4


# =========================
# Helpers
# =========================
def ts(day: str, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hm}")

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    m = np.median(x)
    return float(np.median(np.abs(x - m)) + 1e-6)

def robust_percentile(x: np.ndarray, q: float, default: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.nanpercentile(x, q))

def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371000.0
    latr = math.radians(lat)
    lat0r = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R * dlon * math.cos(0.5 * (latr + lat0r))
    y = R * dlat
    return x, y

def xy_to_latlon(X: np.ndarray, Y: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    R = 6371000.0
    lat0r = np.deg2rad(lat0)
    lat = lat0 + np.rad2deg(Y / R)
    lon = lon0 + np.rad2deg(X / (R * np.cos(lat0r)))
    return lat, lon

def offset_latlon_m(lat: float, lon: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    """Offset a lat/lon by dx (east), dy (north) meters using a local tangent-plane approximation."""
    R = 6371000.0
    lat0r = math.radians(lat)
    dlat = (dy_m / R) * (180.0 / math.pi)
    dlon = (dx_m / (R * max(1e-12, math.cos(lat0r)))) * (180.0 / math.pi)
    return lat + dlat, lon + dlon

def met_from_to_theta(wd_from_deg: float) -> float:
    wd_to = (wd_from_deg + 180.0) % 360.0
    return float(np.deg2rad(wd_to))


# =========================
# IO
# =========================
def read_sensor_csv_1hz(path: str, anchor_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]

    if "Counter" not in df.columns:
        raise ValueError(f"{path}: missing Counter column")

    pm_col = None
    for c in ["PM2.5_Env", "PM2.5_Std", "PM2.5"]:
        if c in df.columns:
            pm_col = c
            break
    if pm_col is None:
        raise ValueError(f"{path}: missing PM2.5 column (PM2.5_Env/PM2.5_Std/PM2.5)")

    df["Counter"] = pd.to_numeric(df["Counter"], errors="coerce")
    df[pm_col]    = pd.to_numeric(df[pm_col], errors="coerce")
    df = df.dropna(subset=["Counter", pm_col]).copy()
    df["Counter"] = df["Counter"].astype(int)

    df["dt"] = anchor_start + pd.to_timedelta(df["Counter"], unit="s")
    df = df.rename(columns={pm_col: "pm25"})[["dt", "pm25"]].sort_values("dt")
    df = df.groupby("dt", as_index=False)["pm25"].mean()
    return df

def load_all_sensors_1min(t_anchor: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    series: Dict[str, pd.Series] = {}
    for fp in sorted(glob.glob(SENSOR_GLOB)):
        base = os.path.basename(fp)
        m = re.search(r"Data_(\d+)\.csv$", base, flags=re.IGNORECASE)
        if not m:
            continue
        sid = f"S{int(m.group(1))}"
        if sid not in SENSOR_LATLON:
            continue

        df = read_sensor_csv_1hz(fp, anchor_start=t_anchor)
        df = df[(df["dt"] >= t_anchor) & (df["dt"] <= t_end)].copy()
        s = df.set_index("dt")["pm25"].resample("1min").mean()
        series[sid] = s

    if not series:
        raise RuntimeError(f"No sensor files matched '{SENSOR_GLOB}' with IDs in SENSOR_LATLON.")

    wide = pd.concat(series, axis=1).sort_index()
    full_idx = pd.date_range(t_anchor, t_end, freq="1min")
    wide = wide.reindex(full_idx).interpolate(limit=3).ffill().bfill()
    return wide

def read_wind_1min_noaa(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(WIND_CSV, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["DATE", "wind_speed", "wind_direction"]:
        if col not in df.columns:
            raise ValueError("Wind CSV must have DATE, wind_speed, wind_direction")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["wind_direction"] = pd.to_numeric(df["wind_direction"], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()
    df = df[(df["DATE"] >= t0) & (df["DATE"] <= t1)].copy()
    df = df.sort_values("DATE").set_index("DATE")

    wd = df["wind_direction"].to_numpy(dtype=float)
    wd[(wd == 999) | (wd < 0) | (wd > 360)] = np.nan
    wd[wd == 360] = 0.0
    df["wind_dir_from"] = wd

    out = df[["wind_speed", "wind_dir_from"]].resample("1min").mean()
    out["wind_speed"] = out["wind_speed"].interpolate(limit=6).ffill().bfill()
    out["wind_dir_from"] = out["wind_dir_from"].interpolate(limit=6).ffill().bfill()
    return out

def smooth_wind(df: pd.DataFrame, win_min: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["wind_speed"] = out["wind_speed"].rolling(win_min, min_periods=1, center=True).mean()

    wd_from = out["wind_dir_from"].to_numpy(dtype=float)
    theta_to = np.array([met_from_to_theta(x) if np.isfinite(x) else np.nan for x in wd_from])
    ux = np.sin(theta_to)
    uy = np.cos(theta_to)

    ux_s = pd.Series(ux, index=out.index).rolling(win_min, min_periods=1, center=True).mean().to_numpy()
    uy_s = pd.Series(uy, index=out.index).rolling(win_min, min_periods=1, center=True).mean().to_numpy()

    theta_to_s = np.arctan2(ux_s, uy_s)
    wd_to_s = (np.rad2deg(theta_to_s) + 360) % 360
    out["wind_dir_from"] = (wd_to_s - 180) % 360
    return out


# =========================
# IDW (used for folium weighting only)
# =========================
def precompute_idw_weights(points_xy: np.ndarray, sensor_xy: np.ndarray, power: float, eps: float) -> np.ndarray:
    dx = points_xy[:, [0]] - sensor_xy[:, 0][None, :]
    dy = points_xy[:, [1]] - sensor_xy[:, 1][None, :]
    d = np.sqrt(dx*dx + dy*dy) + eps
    w = 1.0 / (d ** power)
    wsum = np.sum(w, axis=1, keepdims=True)
    return w / np.maximum(wsum, 1e-12)

def idw_apply(W: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return W @ vals


# =========================
# Centered plume kernel (NO drift)
# =========================
def puff_sigma(age_s: float, sigma0: float, K: float) -> float:
    return float(sigma0 + math.sqrt(max(0.0, 2.0 * K * age_s)))

def centered_kernel(points_xy: np.ndarray,
                    age_s: float,
                    sigma0: float, K: float, tau: float,
                    wind_speed: float, wind_dir_from: float) -> np.ndarray:
    if age_s <= 0:
        return np.zeros(points_xy.shape[0], dtype=float)

    decay = math.exp(-age_s / max(1e-6, tau))
    sigma_perp = puff_sigma(age_s, sigma0, K)

    if PLUME_MODE == "isotropic":
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        denom = 2.0 * math.pi * sigma_perp * sigma_perp
        expo = -0.5 * ((x / sigma_perp) ** 2 + (y / sigma_perp) ** 2)
        return decay * (np.exp(expo) / max(1e-12, denom))

    theta_to = met_from_to_theta(float(wind_dir_from))
    U = max(0.0, float(wind_speed))
    stretch = 1.0 + (U * age_s) / max(1e-6, 2500.0)
    stretch = min(stretch, STRETCH_CAP)
    sigma_par = sigma_perp * stretch

    ux = math.sin(theta_to)
    uy = math.cos(theta_to)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    s = x * ux + y * uy
    n = -x * uy + y * ux

    denom = 2.0 * math.pi * sigma_par * sigma_perp
    expo = -0.5 * ((s / sigma_par) ** 2 + (n / sigma_perp) ** 2)
    return decay * (np.exp(expo) / max(1e-12, denom))

def release_indices(times: pd.DatetimeIndex, fw0: pd.Timestamp, fw1: pd.Timestamp) -> np.ndarray:
    return np.where((times >= fw0) & (times < fw1))[0]


# =========================
# Robust weights + inversion
# =========================
@dataclass(frozen=True)
class BestFit:
    sigma0: float
    K: float
    tau: float
    rmse: float
    Q_profile: np.ndarray
    release_idx: np.ndarray

def compute_sensor_weights(y_resid: np.ndarray, sensors: List[str]) -> pd.Series:
    Ns = y_resid.shape[1]
    scales = np.zeros(Ns, dtype=float)
    for k in range(Ns):
        scales[k] = mad(y_resid[:, k])
    w = 1.0 / np.maximum(scales, 1e-6)
    w = w / np.maximum(np.mean(w), 1e-12)
    w = np.clip(w, WEIGHT_CAP_MIN, WEIGHT_CAP_MAX)
    return pd.Series(w, index=sensors, name="weight")

def invert_Q_profile_weighted(y_ts_ns: np.ndarray,
                              times: pd.DatetimeIndex,
                              rel_idx: np.ndarray,
                              sensor_xy: np.ndarray,
                              wind_speed: np.ndarray,
                              wind_dir_from: np.ndarray,
                              sigma0: float, K: float, tau: float,
                              lam_smooth: float,
                              sensor_weights: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    T, Ns = y_ts_ns.shape
    M = len(rel_idx)
    if M == 0:
        y_pred = np.zeros_like(y_ts_ns)
        return np.zeros(0, dtype=float), float(np.sqrt(np.mean(y_ts_ns**2))), y_pred

    A = np.zeros((T * Ns, M), dtype=float)

    U_rel = wind_speed[rel_idx].astype(float)
    WD_rel = wind_dir_from[rel_idx].astype(float)

    for j in range(M):
        t_rel = times[rel_idx[j]]
        for i in range(rel_idx[j], T):
            age_s = float((times[i] - t_rel).total_seconds())
            kvec = centered_kernel(sensor_xy, age_s, sigma0, K, tau, U_rel[j], WD_rel[j])
            A[i*Ns:(i+1)*Ns, j] = kvec

    y = y_ts_ns.reshape(-1)

    # apply per-sensor weights across all times
    w_s = sensor_weights.astype(float).reshape(1, Ns)
    w_rep = np.tile(w_s, (T, 1)).reshape(-1)
    sqrtw = np.sqrt(np.maximum(w_rep, 1e-12))
    Aw = A * sqrtw[:, None]
    yw = y * sqrtw

    # smoothness
    if M >= 2:
        D = np.zeros((M - 1, M), dtype=float)
        for i in range(M - 1):
            D[i, i] = -1.0
            D[i, i+1] = 1.0
    else:
        D = np.zeros((0, M), dtype=float)

    fro2 = float(np.linalg.norm(Aw, "fro") ** 2)
    lam = float(lam_smooth) * (fro2 / max(1.0, M))

    if D.shape[0] > 0:
        A_aug = np.vstack([Aw, math.sqrt(lam) * D])
        y_aug = np.concatenate([yw, np.zeros(D.shape[0], dtype=float)])
    else:
        A_aug = Aw
        y_aug = yw

    res = lsq_linear(A_aug, y_aug, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto", verbose=0)
    q = res.x.astype(float)

    y_pred = (A @ q).reshape(T, Ns)
    e = rmse(y_ts_ns, y_pred)
    return q, e, y_pred

def grid_search_inversion_weighted(y_resid: np.ndarray,
                                   times: pd.DatetimeIndex,
                                   rel_idx: np.ndarray,
                                   sensor_xy: np.ndarray,
                                   wind_speed: np.ndarray,
                                   wind_dir_from: np.ndarray,
                                   sensor_weights: np.ndarray) -> BestFit:
    best: Optional[BestFit] = None
    for sigma0 in SIGMA0_GRID:
        for K in K_GRID:
            for tau in TAU_GRID:
                q, e, _ = invert_Q_profile_weighted(
                    y_resid, times, rel_idx, sensor_xy,
                    wind_speed, wind_dir_from,
                    float(sigma0), float(K), float(tau),
                    lam_smooth=LAMBDA_SMOOTH,
                    sensor_weights=sensor_weights
                )
                cand = BestFit(float(sigma0), float(K), float(tau), float(e), q, rel_idx)
                if (best is None) or (cand.rmse < best.rmse):
                    best = cand
    assert best is not None
    return best


# =========================
# Frame builder (plume-only enhancement)
# =========================
def build_frames_plume_only(times: pd.DatetimeIndex,
                            wind_full: pd.DataFrame,
                            sensors: List[str],
                            baselines: pd.Series,
                            best: BestFit,
                            half_width_m: float,
                            sensor_xy: Optional[np.ndarray] = None,
                            residual_ts_ns: Optional[np.ndarray] = None,
                            obs_enh_ts_ns: Optional[np.ndarray] = None,
                            fw1: Optional[pd.Timestamp] = None):
    lat0, lon0 = SRC_LAT, SRC_LON

    xs = np.linspace(-half_width_m, half_width_m, GRID_N)
    ys = np.linspace(-half_width_m, half_width_m, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    lat_grid, lon_grid = xy_to_latlon(X, Y, lat0, lon0)

    N = X.shape[0]
    total = N * N
    step = max(1, int(math.sqrt(total / max(1, TARGET_POINTS_PER_FRAME))))
    ii = np.arange(0, N, step)
    jj = np.arange(0, N, step)
    ny, nx = len(ii), len(jj)

    pts_xy = np.stack([X[np.ix_(ii, jj)].ravel(), Y[np.ix_(ii, jj)].ravel()], axis=1)
    pts_lat = lat_grid[np.ix_(ii, jj)].ravel()
    pts_lon = lon_grid[np.ix_(ii, jj)].ravel()

    lon_axis = lon_grid[0, jj].astype(float)
    lat_axis = lat_grid[ii, 0].astype(float)

    # sensor XY in meters around source (used for optional residual assimilation)
    if sensor_xy is None:
        sensor_xy = np.array([latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], lat0, lon0) for s in sensors], dtype=float)

    # Optional residual assimilation weights (Gaussian in meters)
    if ENABLE_ASSIMILATION and (residual_ts_ns is not None) and (ASSIM_ALPHA > 0.0):
        d2 = (pts_xy[:, 0:1] - sensor_xy[None, :, 0])**2 + (pts_xy[:, 1:2] - sensor_xy[None, :, 1])**2
        Wg = np.exp(-0.5 * d2 / (ASSIM_SIGMA_M**2))
        sumWg = Wg.sum(axis=1)
    else:
        Wg = None
        sumWg = None

    # Optional sensor imprint weights (Gaussian in meters; non-normalized)
    if ENABLE_SENSOR_IMPRINT and (obs_enh_ts_ns is not None) and (IMPRINT_ALPHA > 0.0):
        d2i = (pts_xy[:, 0:1] - sensor_xy[None, :, 0])**2 + (pts_xy[:, 1:2] - sensor_xy[None, :, 1])**2
        Wi = np.exp(-0.5 * d2i / (IMPRINT_SIGMA_M**2))
    else:
        Wi = None

    # wind arrays
    ws = wind_full["wind_speed"].to_numpy(dtype=float)
    wd = wind_full["wind_dir_from"].to_numpy(dtype=float)

    rel_idx = best.release_idx
    q_prof = best.Q_profile
    rng = np.random.default_rng(int(SENSOR_CLOUD_SEED)) if ADD_SENSOR_CLOUD else None

    U_rel = ws[rel_idx].astype(float)
    WD_rel = wd[rel_idx].astype(float)
    t_rel = times[rel_idx]

    z_frames: List[np.ndarray] = []
    folium_frames_raw: List[List[List[float]]] = []
    max_enh = 0.0

    for i, t in enumerate(times):
        plume = np.zeros(pts_xy.shape[0], dtype=float)
        for j in range(len(rel_idx)):
            if rel_idx[j] > i:
                break
            age_s = float((t - t_rel[j]).total_seconds())
            plume += q_prof[j] * centered_kernel(
                pts_xy, age_s,
                best.sigma0, best.K, best.tau,
                U_rel[j], WD_rel[j]
            )

        enh = plume  # enhancement above baseline

        # Sensor imprint: add local 'glow' near sensors based on observed enhancement (helps sparse grids)
        if Wi is not None and (obs_enh_ts_ns is not None):
            o = obs_enh_ts_ns[i, :].astype(float)
            if IMPRINT_MIN_ENH is not None and IMPRINT_MIN_ENH > 0:
                o = np.where(o >= IMPRINT_MIN_ENH, o, 0.0)
            imp = Wi @ o
            if IMPRINT_CLIP is not None and IMPRINT_CLIP > 0:
                imp = np.clip(imp, 0.0, IMPRINT_CLIP)
            enh = enh + IMPRINT_ALPHA * imp
        # Folium sensor cloud: add extra points around sensors so each sensor shows color even if grid is sparse
        if ADD_SENSOR_CLOUD and (obs_enh_ts_ns is not None) and (rng is not None):
            obs_v = obs_enh_ts_ns[i, :].astype(float)
            for k, sname in enumerate(sensors):
                val = float(obs_v[k]) * float(SENSOR_CLOUD_SCALE)
                if (not np.isfinite(val)) or (val < IMPRINT_MIN_ENH):
                    continue
                slat, slon = SENSOR_LATLON[sname]
                for _ in range(int(SENSOR_CLOUD_N)):
                    rr = float(SENSOR_CLOUD_RADIUS_M) * math.sqrt(float(rng.random()))
                    ang = 2.0 * math.pi * float(rng.random())
                    dx = rr * math.cos(ang)
                    dy = rr * math.sin(ang)
                    latp, lonp = offset_latlon_m(slat, slon, dx, dy)
                    pts.append([float(latp), float(lonp), float(val)])


        # Extra post-fireworks fade for clearer decay (display physics stays center-locked; this is just fade)
        if (POST_EVENT_DECAY_TAU is not None) and (POST_EVENT_DECAY_TAU > 0) and (fw1 is not None) and (t > fw1):
            fade = math.exp(-float((t - fw1).total_seconds()) / float(POST_EVENT_DECAY_TAU))
            enh = enh * fade

        enh = np.clip(enh, 0.0, np.inf)

        # Optional residual assimilation: locally nudge field toward sensor observations (without moving the source)
        if Wg is not None:
            r = residual_ts_ns[i, :].astype(float)
            corr = (Wg @ r) / (sumWg + 1e-12)
            corr[sumWg < ASSIM_MIN_WEIGHT] = 0.0
            if ASSIM_CLIP is not None and ASSIM_CLIP > 0:
                corr = np.clip(corr, -ASSIM_CLIP, ASSIM_CLIP)
            enh = np.clip(enh + ASSIM_ALPHA * corr, 0.0, np.inf)


        if np.any(np.isfinite(enh)):
            max_enh = max(max_enh, float(np.nanmax(enh)))

        z = enh.reshape(ny, nx).astype(np.float32)
        z_frames.append(z)

        pts = [[float(pts_lat[p]), float(pts_lon[p]), float(enh[p])]
               for p in range(enh.size) if np.isfinite(enh[p])]
        if not pts:
            pts = [[float(SRC_LAT), float(SRC_LON), 0.0]]
        folium_frames_raw.append(pts)

    return lon_axis, lat_axis, z_frames, folium_frames_raw, max_enh


# =========================
# Save Plotly / Folium
# =========================
def plotly_colorscale_transparent0():
    return [
        [0.0,  "rgba(0,0,0,0)"],
        [0.05, "rgba(0, 90, 255, 0.25)"],
        [0.25, "rgba(0, 200, 255, 0.55)"],
        [0.50, "rgba(255, 255, 0, 0.75)"],
        [1.00, "rgba(255, 0, 0, 0.90)"],
    ]

def save_plotly(lon_axis, lat_axis, z_frames, labels,
                sensors: List[str],
                obs_enh: np.ndarray,
                cmax: float, zmin_display: float,
                out_html: str):
    if not _HAVE_PLOTLY:
        print("Plotly not installed. Run: pip install plotly")
        return

    # transparency
    z_frames2 = []
    for z in z_frames:
        zz = z.copy()
        zz[zz < zmin_display] = np.nan
        z_frames2.append(zz)

    s_lats = [SENSOR_LATLON[s][0] for s in sensors]
    s_lons = [SENSOR_LATLON[s][1] for s in sensors]
    cmax_s = max(1e-6, robust_percentile(obs_enh, 99.0, 10.0))

    sensor_trace = go.Scatter(
        x=s_lons, y=s_lats,
        mode="markers+text",
        text=sensors,
        textposition="top center",
        marker=dict(
            size=14,
            color=obs_enh[0, :] if SHOW_SENSOR_OVERLAY else None,
            cmin=0, cmax=cmax_s,
            colorscale="Turbo",
            colorbar=dict(title="Sensor Enh (µg/m³)") if SHOW_SENSOR_OVERLAY else None,
            line=dict(width=1)
        ),
        name="Sensors (obs enh)"
    )

    src_trace = go.Scatter(
        x=[SRC_LON], y=[SRC_LAT],
        mode="markers+text",
        text=["Source"],
        textposition="bottom center",
        marker=dict(size=14, symbol="x"),
        name="Source"
    )

    hm0 = go.Heatmap(
        z=z_frames2[0], x=lon_axis, y=lat_axis,
        colorscale=plotly_colorscale_transparent0(),
        zmin=0.0, zmax=float(cmax),
        colorbar=dict(title="Enhancement (µg/m³)"),
        name="Field"
    )

    frames = []
    for k in range(len(z_frames2)):
        fr_data = [
            go.Heatmap(
                z=z_frames2[k], x=lon_axis, y=lat_axis,
                colorscale=plotly_colorscale_transparent0(),
                zmin=0.0, zmax=float(cmax),
                showscale=False
            ),
        ]
        if SHOW_SENSOR_OVERLAY:
            fr_data.append(go.Scatter(
                x=s_lons, y=s_lats,
                mode="markers+text",
                text=sensors,
                textposition="top center",
                marker=dict(
                    size=14,
                    color=obs_enh[k, :],
                    cmin=0, cmax=cmax_s,
                    colorscale="Turbo",
                    showscale=False,
                    line=dict(width=1)
                )
            ))
        else:
            fr_data.append(go.Scatter(x=s_lons, y=s_lats, mode="markers+text", text=sensors, textposition="top center"))

        fr_data.append(go.Scatter(
            x=[SRC_LON], y=[SRC_LAT],
            mode="markers+text",
            text=["Source"],
            textposition="bottom center",
            marker=dict(size=14, symbol="x")
        ))
        frames.append(go.Frame(data=fr_data, name=str(k), layout=go.Layout(title=f"Baseline-only plume — {labels[k]}")))

    steps = []
    for k in range(len(labels)):
        steps.append(dict(
            method="animate",
            args=[[str(k)], {"mode": "immediate", "frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
            label=labels[k][-5:],
        ))

    fig = go.Figure(data=[hm0, sensor_trace, src_trace], frames=frames)
    fig.update_layout(
        title=f"Baseline-only centered plume — {labels[0]}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=1020, height=860,
        template="plotly_white",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.01, y=1.06,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            x=0.01, y=0.02, len=0.98,
            steps=steps,
            currentvalue=dict(prefix="Time: ", visible=True),
        )]
    )
    fig.write_html(out_html, include_plotlyjs="inline", full_html=True)
    print("Saved:", out_html)

def save_folium(frames_raw, labels, cmax: float, out_html: str):
    if not _HAVE_FOLIUM:
        print("Folium not installed. Run: pip install folium")
        return

    frames = []
    for fr in frames_raw:
        pts = []
        for lat, lon, val in fr:
            w = float(np.clip(val, 0.0, cmax) / max(1e-6, cmax))
            if w >= FOLIUM_WEIGHT_MIN_KEEP:
                pts.append([lat, lon, w])
        if not pts:
            pts = [[SRC_LAT, SRC_LON, 1e-6]]
        frames.append(pts)

    m = folium.Map(location=(SRC_LAT, SRC_LON), zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.Marker((SRC_LAT, SRC_LON), popup="Source").add_to(m)
    for sid, (slat, slon) in SENSOR_LATLON.items():
        folium.CircleMarker((slat, slon), radius=6, popup=sid, fill=True).add_to(m)

    # HeatMapWithTime API differs across folium versions. Some versions do not accept blur.
    try:
        HeatMapWithTime(
            frames,
            index=labels,
            radius=int(FOLIUM_RADIUS),
            blur=float(FOLIUM_BLUR),
            min_opacity=0.25,
            max_opacity=0.95,
            use_local_extrema=False,
            auto_play=False
        ).add_to(m)
    except TypeError:
        HeatMapWithTime(
            frames,
            index=labels,
            radius=int(FOLIUM_RADIUS),
            min_opacity=0.25,
            max_opacity=0.95,
            use_local_extrema=False,
            auto_play=False
        ).add_to(m)

    m.save(out_html)
    print("Saved:", out_html)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t_anchor = ts(EVENT_DATE, DEVICE_ON_TIME)
    t0 = ts(EVENT_DATE, ANALYSIS_START)
    t1 = ts(EVENT_DATE, END_TIME)

    b0 = ts(EVENT_DATE, BASELINE_START)
    b1 = ts(EVENT_DATE, BASELINE_END)

    fw0 = ts(EVENT_DATE, FIREWORKS_START)
    fw1 = ts(EVENT_DATE, FIREWORKS_END)

    # Load data (anchor -> end)
    wide_all = load_all_sensors_1min(t_anchor, t1)
    wind_all = smooth_wind(read_wind_1min_noaa(t_anchor, t1), win_min=5)

    # align times
    idx = wide_all.index.intersection(wind_all.index)
    wide_all = wide_all.loc[idx]
    wind_all = wind_all.loc[idx]

    # analysis window
    wide = wide_all.loc[wide_all.index >= t0].copy()
    wind = wind_all.loc[wind_all.index >= t0].copy()

    sensors_all = list(wide.columns)
    sensors = [s for s in sensors_all if s not in FORCE_EXCLUDE_SENSORS]
    if not sensors:
        raise RuntimeError("All sensors excluded. Please adjust FORCE_EXCLUDE_SENSORS.")
    print("Loaded sensors:", sensors_all)
    print("Using sensors:", sensors, "(excluded:", [s for s in sensors_all if s not in sensors], ")")

    # build uniform timeline (minute)
    times = pd.date_range(t0, t1, freq=f"{DT_MIN}min")
    wide_fit = wide.reindex(times).interpolate(limit=3).ffill().bfill()
    wind_fit = wind.reindex(times).interpolate(limit=6).ffill().bfill()

    # baseline (YOUR requirement: median in [22:15, 22:20))
    base_win = wide_fit.loc[(wide_fit.index >= b0) & (wide_fit.index < b1), sensors].copy()
    if base_win.empty:
        # fallback: first 5 minutes of analysis
        base_win = wide_fit[sensors].iloc[:5].copy()
    baselines = base_win.median(axis=0, skipna=True)
    baselines.to_csv(os.path.join(OUT_DIR, "baseline_values.csv"), header=["baseline_pm25"])
    print("Saved:", os.path.join(OUT_DIR, "baseline_values.csv"))
    print(f"Baseline window = [{b0.strftime('%H:%M')}, {b1.strftime('%H:%M')})")

    # enhancement above baseline (for inversion and sensor overlay)
    enh = (wide_fit[sensors] - baselines[sensors]).clip(lower=0.0)
    y_resid = enh.to_numpy(dtype=float)

    # sensor XY in meters around source
    lat0, lon0 = SRC_LAT, SRC_LON
    sensor_xy = np.array([latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], lat0, lon0) for s in sensors], dtype=float)

    # release indices
    rel_idx = release_indices(times, fw0, fw1)
    if len(rel_idx) == 0:
        raise RuntimeError("Fireworks window has no samples. Check FIREWORKS_START/END and ANALYSIS_START/END.")

    # wind arrays
    ws = wind_fit["wind_speed"].to_numpy(dtype=float)
    wd = wind_fit["wind_dir_from"].to_numpy(dtype=float)

    # robust sensor weights
    w_series = compute_sensor_weights(y_resid, sensors)
    w_series.to_csv(os.path.join(OUT_DIR, "sensor_weights.csv"))
    print("Saved:", os.path.join(OUT_DIR, "sensor_weights.csv"))
    print("Sensor weights:", w_series.to_dict())

    # inversion grid search
    best = grid_search_inversion_weighted(
        y_resid=y_resid,
        times=times,
        rel_idx=rel_idx,
        sensor_xy=sensor_xy,
        wind_speed=ws,
        wind_dir_from=wd,
        sensor_weights=w_series.to_numpy(dtype=float),
    )
    print("=== BEST v4b (baseline-only enhancement inversion) ===")
    print({
        "sigma0": best.sigma0,
        "K": best.K,
        "tau": best.tau,
        "rmse_enh": best.rmse,
        "Q_len": int(len(best.Q_profile)),
        "PLUME_MODE": PLUME_MODE,
        "STRETCH_CAP": STRETCH_CAP
    })

    # refit best to get y_pred at sensors (diagnostic)
    q, _, y_pred = invert_Q_profile_weighted(
        y_resid, times, rel_idx, sensor_xy, ws, wd,
        best.sigma0, best.K, best.tau,
        lam_smooth=LAMBDA_SMOOTH,
        sensor_weights=w_series.to_numpy(dtype=float)
    )
    best = replace(best, Q_profile=q)  # keep frames consistent with diagnostic refit
    residual_ts_ns = (y_resid - y_pred).astype(float)


    # auto domain sizing
    r_sensors = float(np.max(np.sqrt(sensor_xy[:, 0]**2 + sensor_xy[:, 1]**2)))
    age_max = float((t1 - fw0).total_seconds())
    sigma_max = puff_sigma(max(60.0, age_max), best.sigma0, best.K)
    r_diff = 6.0 * sigma_max
    half_width = max(MIN_HALF_WIDTH_M, r_sensors + 800.0, r_diff)
    half_width = float(np.clip(half_width, MIN_HALF_WIDTH_M, MAX_HALF_WIDTH_M))
    if not AUTO_DOMAIN:
        half_width = float(MIN_HALF_WIDTH_M)
    print(f"Domain half-width = {half_width:.0f} m (AUTO_DOMAIN={AUTO_DOMAIN})")

    # build frames (plume-only)
    lon_axis, lat_axis, z_frames, folium_frames_raw, max_enh = build_frames_plume_only(
        times=times,
        wind_full=wind_fit,
        sensors=sensors,
        baselines=baselines,
        best=best,
        half_width_m=half_width,
        sensor_xy=sensor_xy,
        residual_ts_ns=residual_ts_ns,
        obs_enh_ts_ns=y_resid,
        fw1=fw1
    )

    # visualization scaling
    flat_all = np.concatenate([z.ravel() for z in z_frames])
    cmax = robust_percentile(flat_all, 99.5, DEFAULT_CMAX) if AUTO_CMAX else float(DEFAULT_CMAX)
    cmax = max(5.0, float(cmax))
    zmin = max(DEFAULT_ZMIN, 0.02 * cmax) if AUTO_ZMIN else float(DEFAULT_ZMIN)
    print(f"DEBUG: max_enh_field≈{max_enh:.2f} µg/m³, CMAX={cmax:.2f}, Z_MIN_DISPLAY={zmin:.3f}")

    labels = [t.strftime("%Y-%m-%d %H:%M") for t in times]

    # save Plotly
    obs_enh = enh.to_numpy(dtype=float)  # overlay
    out_plotly = os.path.join(OUT_DIR, "centered_baseline_plume_plotly.html")
    save_plotly(lon_axis, lat_axis, z_frames, labels, sensors, obs_enh, cmax, zmin, out_plotly)

    # save Folium
    out_folium = os.path.join(OUT_DIR, "centered_baseline_plume_folium.html")
    save_folium(folium_frames_raw, labels, cmax, out_folium)

    # emission profile outputs
    t_rel = times[best.release_idx]
    qdf = pd.DataFrame({"time": t_rel, "Q": best.Q_profile})
    qdf.to_csv(os.path.join(OUT_DIR, "emission_profile.csv"), index=False)

    plt.figure(figsize=(10, 3.2))
    plt.plot(t_rel, best.Q_profile, marker="o", lw=1.2)
    plt.title("Inverted emission profile Q(t) (baseline-only enhancement)")
    plt.ylabel("Q(t) (a.u.)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.25)
    out_q = os.path.join(OUT_DIR, "emission_profile.png")
    plt.tight_layout()
    plt.savefig(out_q, dpi=220)
    plt.close()
    print("Saved:", out_q, "and emission_profile.csv")

    # max field vs time
    max_series = np.array([np.nanmax(z) if np.any(np.isfinite(z)) else 0.0 for z in z_frames], dtype=float)
    plt.figure(figsize=(10, 3.2))
    plt.plot(times, max_series, lw=1.2)
    plt.axvspan(fw0, fw1, alpha=0.15)
    plt.title("Max enhancement vs time (baseline-only plume display)")
    plt.ylabel("Max enhancement (µg/m³)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.25)
    out_m = os.path.join(OUT_DIR, "field_max_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_m, dpi=220)
    plt.close()
    print("Saved:", out_m)

    # sensor fit diagnostics: obs enh vs model enh
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 2.3*len(sensors)), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    for k, s in enumerate(sensors):
        ax = axes[k]
        ax.plot(times, y_resid[:, k], lw=1.0, label=f"{s} obs enh = max(0,PM-baseline)")
        ax.plot(times, y_pred[:, k], lw=1.0, label=f"{s} model enh (plume)")
        ax.axvspan(fw0, fw1, alpha=0.15)
        ax.set_ylabel("Enh (µg/m³)")
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Sensor enhancement fit: Observed vs Model (baseline-only inversion)")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "sensor_fit_enhancement.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved:", out_png)

    print("\\nView:")
    print("  python -m http.server 8000")
    print(f"  http://localhost:8000/{OUT_DIR}/centered_baseline_plume_plotly.html")
    print(f"  http://localhost:8000/{OUT_DIR}/centered_baseline_plume_folium.html")

    print("\\nTips:")
    print("  - If the plume stays too long, reduce TAU_GRID (e.g. [300,600,900,1200]).")
    print("  - If wind dominates shape too much, set STRETCH_CAP=1.2 or PLUME_MODE='isotropic'.")
    print("  - If a sensor dominates inversion, add it to FORCE_EXCLUDE_SENSORS or keep robust weights (already on).")


if __name__ == "__main__":
    main()
