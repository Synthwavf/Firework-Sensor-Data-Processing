#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s_calc_inverse_plume.py

Approach 1: "S-calc" inverse point-source plume (effective source strength).

Summary:
- Use wind direction variability to build a virtual crosswind transect at each sensor.
- Bin concentration by wind direction (default 2 deg bins).
- Fit a Gaussian crosswind profile C(y) to estimate sigma_y and centerline C0.
- Convert C0 to an effective source strength S_eff using a simple plume formula.

Notes:
- This is a near-field method; fireworks are not a steady point source.
- S_eff is an effective proxy, not a true emission rate.

Outputs (OUT_DIR):
- s_calc_summary.csv (per window, per sensor estimates)
- s_calc_profiles.csv (bin-level profiles)
- s_calc_timeseries.png (S_eff time series by sensor)

Run:
  python s_calc_inverse_plume.py
"""

from __future__ import annotations

import os
import re
import glob
import math
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG (edit these)
# =========================
EVENT_DATE = "2025-07-04"

DEVICE_ON_TIME = "20:00"     # Counter=0 anchor time
ANALYSIS_START = "22:15"
END_TIME       = "23:30"

# Baseline window (used if BACKGROUND_METHOD="baseline_window")
BASELINE_START = "22:15"
BASELINE_END   = "22:20"

SENSOR_GLOB = "Data_*.csv"
WIND_CSV    = "wind_speed_2025.csv"
OUT_DIR     = "s_calc_outputs"

# Source (fixed)
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

DT_MIN = 1

# S-calc windowing
WINDOW_MIN = 20
STEP_MIN = 5

# Background definition
BACKGROUND_METHOD = "baseline_window"   # "baseline_window" or "percentile"
BACKGROUND_PERCENTILE = 10.0

# Wind direction binning
BIN_DEG = 2.0
MAX_ABS_ANGLE = 85.0
MIN_SAMPLES_PER_BIN = 5
MIN_BINS_FOR_FIT = 6

# Filters
MIN_WIND_SPEED = 0.5
MIN_DOWNWIND_M = 10.0

# Gaussian conversion
SIGMA_Z_RATIO = 0.6     # sigma_z = SIGMA_Z_RATIO * sigma_y
MIN_SIGMA_Y = 5.0
MAX_SIGMA_Y = 4000.0


# =========================
# Helpers
# =========================
def ts(day: str, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hm}")

def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371000.0
    latr = math.radians(lat)
    lat0r = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R * dlon * math.cos(0.5 * (latr + lat0r))
    y = R * dlat
    return x, y

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
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
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
# S-calc core
# =========================
def compute_background(series: pd.Series, baseline: float) -> float:
    if BACKGROUND_METHOD == "baseline_window":
        return float(baseline)
    return float(np.nanpercentile(series.to_numpy(dtype=float), BACKGROUND_PERCENTILE))

def compute_crosswind_samples(r_xy: Tuple[float, float],
                              wind_speed: np.ndarray,
                              wind_dir_from: np.ndarray,
                              conc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_x, r_y = r_xy
    theta = np.array([met_from_to_theta(wd) if np.isfinite(wd) else np.nan for wd in wind_dir_from], dtype=float)
    ux = np.sin(theta)
    uy = np.cos(theta)
    x = r_x * ux + r_y * uy
    y = -r_x * uy + r_y * ux
    angle = np.rad2deg(np.arctan2(y, x))

    mask = np.isfinite(conc) & np.isfinite(wind_speed) & np.isfinite(angle)
    mask &= (wind_speed >= MIN_WIND_SPEED)
    mask &= (x >= MIN_DOWNWIND_M)
    mask &= (np.abs(angle) <= MAX_ABS_ANGLE)
    return angle[mask], y[mask], wind_speed[mask], conc[mask]

def bin_profile(angle_deg: np.ndarray,
                wind_speed: np.ndarray,
                conc: np.ndarray) -> pd.DataFrame:
    if angle_deg.size == 0:
        return pd.DataFrame(columns=["angle_center_deg", "mean_conc", "mean_wind", "n"])

    bin_start = -MAX_ABS_ANGLE
    bin_end = MAX_ABS_ANGLE
    n_bins = int(math.floor((bin_end - bin_start) / BIN_DEG))
    idx = np.floor((angle_deg - bin_start) / BIN_DEG).astype(int)
    valid = (idx >= 0) & (idx < n_bins)

    df = pd.DataFrame({
        "bin": idx[valid],
        "angle_deg": angle_deg[valid],
        "wind": wind_speed[valid],
        "conc": conc[valid]
    })
    if df.empty:
        return pd.DataFrame(columns=["angle_center_deg", "mean_conc", "mean_wind", "n"])

    grp = df.groupby("bin", as_index=False).agg(
        n=("conc", "size"),
        mean_conc=("conc", "mean"),
        mean_wind=("wind", "mean"),
        mean_angle=("angle_deg", "mean")
    )
    grp = grp[grp["n"] >= MIN_SAMPLES_PER_BIN].copy()
    if grp.empty:
        return pd.DataFrame(columns=["angle_center_deg", "y_m", "mean_conc", "mean_wind", "n"])

    grp["angle_center_deg"] = bin_start + (grp["bin"].astype(float) + 0.5) * BIN_DEG
    return grp[["angle_center_deg", "mean_conc", "mean_wind", "n"]].copy()

def fit_profile(profile: pd.DataFrame, range_m: float) -> Tuple[float, float, float, float]:
    if profile.empty or range_m <= 0:
        return np.nan, np.nan, np.nan, np.nan

    ang = profile["angle_center_deg"].to_numpy(dtype=float)
    y = range_m * np.sin(np.deg2rad(ang))
    c = profile["mean_conc"].to_numpy(dtype=float)
    w = profile["n"].to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(c) & (c > 0)
    if mask.sum() < MIN_BINS_FOR_FIT:
        return np.nan, np.nan, np.nan, np.nan

    y2 = y[mask] ** 2
    logc = np.log(c[mask])
    weights = np.sqrt(np.maximum(w[mask], 1.0))
    coef = np.polyfit(y2, logc, deg=1, w=weights)
    slope, intercept = float(coef[0]), float(coef[1])
    if slope >= 0:
        return np.nan, np.nan, np.nan, np.nan

    sigma_y = math.sqrt(-1.0 / (2.0 * slope))
    sigma_y = float(np.clip(sigma_y, MIN_SIGMA_Y, MAX_SIGMA_Y))
    c0 = float(math.exp(intercept))

    logc_hat = intercept + slope * y2
    ss_res = float(np.sum((logc - logc_hat) ** 2))
    ss_tot = float(np.sum((logc - np.mean(logc)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    return sigma_y, c0, r2, float(mask.sum())


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

    wide_all = load_all_sensors_1min(t_anchor, t1)
    wind_all = smooth_wind(read_wind_1min_noaa(t_anchor, t1), win_min=5)

    idx = wide_all.index.intersection(wind_all.index)
    wide_all = wide_all.loc[idx]
    wind_all = wind_all.loc[idx]

    wide = wide_all.loc[(wide_all.index >= t0) & (wide_all.index <= t1)].copy()
    wind = wind_all.loc[(wind_all.index >= t0) & (wind_all.index <= t1)].copy()

    sensors = list(wide.columns)
    print("Loaded sensors:", sensors)

    times = pd.date_range(t0, t1, freq=f"{DT_MIN}min")
    wide_fit = wide.reindex(times).interpolate(limit=3).ffill().bfill()
    wind_fit = wind.reindex(times).interpolate(limit=6).ffill().bfill()

    base_win = wide_fit.loc[(wide_fit.index >= b0) & (wide_fit.index < b1), sensors].copy()
    if base_win.empty:
        base_win = wide_fit[sensors].iloc[:5].copy()
    baselines = base_win.median(axis=0, skipna=True)

    # precompute sensor XY and range
    lat0, lon0 = SRC_LAT, SRC_LON
    sensor_xy = {
        s: latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], lat0, lon0)
        for s in sensors
    }
    sensor_r = {s: float(math.sqrt(x*x + y*y)) for s, (x, y) in sensor_xy.items()}

    # window starts
    win_delta = pd.Timedelta(minutes=WINDOW_MIN)
    step_delta = pd.Timedelta(minutes=STEP_MIN)
    win_starts: List[pd.Timestamp] = []
    cur = t0
    while cur + win_delta <= t1:
        win_starts.append(cur)
        cur += step_delta

    results = []
    profile_rows = []

    for s in sensors:
        r_xy = sensor_xy[s]
        range_m = sensor_r[s]
        baseline_s = baselines[s]

        for t_start in win_starts:
            t_end = t_start + win_delta
            pm = wide_fit.loc[(wide_fit.index >= t_start) & (wide_fit.index < t_end), s]
            wd = wind_fit.loc[(wind_fit.index >= t_start) & (wind_fit.index < t_end), "wind_dir_from"]
            ws = wind_fit.loc[(wind_fit.index >= t_start) & (wind_fit.index < t_end), "wind_speed"]

            if pm.empty or wd.empty or ws.empty:
                continue

            bg = compute_background(pm, baseline_s)
            enh = (pm - bg).clip(lower=0.0).to_numpy(dtype=float)
            angle, y_m, ws_v, enh_v = compute_crosswind_samples(
                r_xy=r_xy,
                wind_speed=ws.to_numpy(dtype=float),
                wind_dir_from=wd.to_numpy(dtype=float),
                conc=enh
            )

            profile = bin_profile(angle, ws_v, enh_v)
            if not profile.empty:
                for _, row in profile.iterrows():
                    profile_rows.append({
                        "sensor": s,
                        "window_start": t_start,
                        "window_end": t_end,
                        "angle_center_deg": float(row["angle_center_deg"]),
                        "y_m": float(range_m * math.sin(math.radians(float(row["angle_center_deg"])))),
                        "mean_conc": float(row["mean_conc"]),
                        "mean_wind": float(row["mean_wind"]),
                        "n_samples": int(row["n"])
                    })

            sigma_y, c0, r2, n_bins_used = fit_profile(profile, range_m=range_m)
            if np.isfinite(sigma_y) and np.isfinite(c0):
                sigma_z = float(SIGMA_Z_RATIO * sigma_y)
                u_mean = float(np.nanmean(ws_v)) if ws_v.size > 0 else np.nan
                s_eff = float(c0 * math.pi * u_mean * sigma_y * sigma_z) if np.isfinite(u_mean) else np.nan
            else:
                sigma_z = np.nan
                u_mean = float(np.nanmean(ws_v)) if ws_v.size > 0 else np.nan
                s_eff = np.nan

            results.append({
                "sensor": s,
                "window_start": t_start,
                "window_end": t_end,
                "n_samples": int(enh_v.size),
                "n_bins": int(profile.shape[0]),
                "u_mean": u_mean,
                "sigma_y_m": sigma_y,
                "sigma_z_m": sigma_z,
                "c0_ug_m3": c0,
                "s_eff_ug_s": s_eff,
                "fit_r2": r2,
                "range_m": range_m,
                "background": bg
            })

    df = pd.DataFrame(results)
    out_summary = os.path.join(OUT_DIR, "s_calc_summary.csv")
    df.to_csv(out_summary, index=False)
    print("Saved:", out_summary)

    df_prof = pd.DataFrame(profile_rows)
    out_prof = os.path.join(OUT_DIR, "s_calc_profiles.csv")
    df_prof.to_csv(out_prof, index=False)
    print("Saved:", out_prof)

    # time series plot
    if not df.empty:
        df["t_center"] = df["window_start"] + (df["window_end"] - df["window_start"]) / 2
        plt.figure(figsize=(10.5, 4.2))
        for s in sensors:
            sub = df[df["sensor"] == s]
            plt.plot(sub["t_center"], sub["s_eff_ug_s"], marker="o", lw=1.0, label=s)
        median_series = df.groupby("t_center")["s_eff_ug_s"].median().sort_index()
        plt.plot(median_series.index, median_series.values, color="k", lw=2.0, label="median")
        plt.title("S-calc effective source strength (per sensor)")
        plt.ylabel("S_eff (ug/s)")
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, frameon=False)
        out_png = os.path.join(OUT_DIR, "s_calc_timeseries.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
        print("Saved:", out_png)
    else:
        print("No valid S-calc windows found; check filters and wind data.")


if __name__ == "__main__":
    main()
