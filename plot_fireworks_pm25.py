#!/usr/bin/env python3
"""
Plot Lake Union fireworks PM2.5 (1 Hz) from multiple low-cost sensors and (optionally)
overlay WA Ecology / regulatory monitor PM2.5 exported from https://airqualitymap.ecology.wa.gov/

Assumptions for your sensor CSVs:
- First row is a device ID (e.g., "A5A5A5A5A5A5") and should be skipped
- Data rows include:
    Counter, Date, Time, ... , PM2.5_Env, ...
- The Date/Time fields may not contain a real calendar date (often "2000/1/1"),
  so we anchor absolute time using --event_date and --start_time and the Counter (seconds).

Example:
  python plot_fireworks_pm25.py --sensor_glob "Data_*.csv" --zoom_max 250 --event_date 2025-07-04 --start_time 20:00 --end_time 23:30 --fireworks_start 22:20 --fireworks_end 22:40 --out_prefix lake_union

Optional public monitor overlay (exported CSV):
  python plot_fireworks_pm25.py --sensor_glob "Data_*.csv" --event_date 2025-07-04 --start_time 20:00 --end_time 23:30 --fireworks_start 22:20 --fireworks_end 22:40 --public_csv ecology_pm25.csv --public_time_col DateTime --public_pm25_col "PM2.5" --public_resample 1min --out_prefix lake_union_with_public

"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class TimeConfig:
    event_date: str          # YYYY-MM-DD
    start_time: str          # HH:MM (24h)
    end_time: str            # HH:MM (24h)
    fireworks_start: str     # HH:MM
    fireworks_end: str       # HH:MM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sensor_glob", required=True,
                   help='Glob for sensor CSVs, e.g. "Data_*.csv"')
    p.add_argument("--event_date", required=True,
                   help="Event date (local): YYYY-MM-DD, e.g. 2025-07-04")
    p.add_argument("--start_time", default="20:00",
                   help="Sensors on time (local): HH:MM (default 20:00)")
    p.add_argument("--end_time", default="23:30",
                   help="Analysis end time (local): HH:MM (default 23:30)")
    p.add_argument("--fireworks_start", default="22:20",
                   help="Fireworks start time (local): HH:MM (default 22:20)")
    p.add_argument("--fireworks_end", default="22:40",
                   help="Fireworks end time (local): HH:MM (default 22:40)")

    # Optional: public / regulatory monitor CSV
    p.add_argument("--public_csv", default=None,
                   help="Optional public monitor CSV to overlay")
    p.add_argument("--public_time_col", default=None,
                   help="Datetime column name in public CSV (required if --public_csv is used)")
    p.add_argument("--public_pm25_col", default=None,
                   help="PM2.5 column name in public CSV (required if --public_csv is used)")
    p.add_argument("--public_resample", default="1min",
                   help='Resample interval for overlay, e.g. "1min", "5min", "1H" (default 1min)')

    # Output
    p.add_argument("--out_prefix", default="pm25",
                   help="Output file prefix (default: pm25)")
    p.add_argument("--zoom_max", type=float, default=120.0,
                   help="Upper y-limit for zoom plot (default 120)")
    return p.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_sensor_csv(path: str, anchor_start: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: dt, pm25 (float), counter (int)
    """
    # Skip the first line (device ID)
    df = pd.read_csv(path, skiprows=1)
    df = _normalize_columns(df)

    # Required columns
    if "Counter" not in df.columns:
        raise ValueError(f"{path}: missing 'Counter' column. Found: {df.columns.tolist()}")
    pm_col = None
    for candidate in ["PM2.5_Env", "PM2.5_Std", "PM2.5"]:
        if candidate in df.columns:
            pm_col = candidate
            break
    if pm_col is None:
        raise ValueError(f"{path}: cannot find a PM2.5 column (PM2.5_Env / PM2.5_Std / PM2.5).")

    # Clean + coerce
    df["Counter"] = pd.to_numeric(df["Counter"], errors="coerce").astype("Int64")
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")

    df = df.dropna(subset=["Counter", pm_col]).copy()
    df["Counter"] = df["Counter"].astype(int)

    # Build datetime from counter seconds
    df["dt"] = anchor_start + pd.to_timedelta(df["Counter"], unit="s")
    df = df.rename(columns={pm_col: "pm25"})
    df = df[["dt", "Counter", "pm25"]].sort_values("dt")
    return df


def build_time_bounds(tc: TimeConfig) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    day = pd.to_datetime(tc.event_date)
    t0 = pd.to_datetime(f"{tc.event_date} {tc.start_time}")
    t1 = pd.to_datetime(f"{tc.event_date} {tc.end_time}")
    fw0 = pd.to_datetime(f"{tc.event_date} {tc.fireworks_start}")
    fw1 = pd.to_datetime(f"{tc.event_date} {tc.fireworks_end}")

    if not (t0 <= fw0 <= fw1 <= t1):
        raise ValueError("Time bounds must satisfy start <= fireworks_start <= fireworks_end <= end")
    return t0, t1, fw0, fw1


def compute_window_stats(wide: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp,
                         fw0: pd.Timestamp, fw1: pd.Timestamp) -> pd.DataFrame:
    """
    wide: index=dt, columns = sensors (pm25)
    Returns stats table for pre / fireworks / post with max/min/mean per sensor + network spread.
    """
    def stats_for_window(name: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
        sub = wide.loc[(wide.index >= a) & (wide.index <= b)]
        s = {}
        # per-sensor peaks
        for c in sub.columns:
            s[f"{c}_max"] = float(np.nanmax(sub[c].values)) if len(sub) else np.nan
            s[f"{c}_min"] = float(np.nanmin(sub[c].values)) if len(sub) else np.nan
            s[f"{c}_mean"] = float(np.nanmean(sub[c].values)) if len(sub) else np.nan
        # network spread
        row_max = sub.max(axis=1, skipna=True)
        row_min = sub.min(axis=1, skipna=True)
        spread = row_max - row_min
        s["network_spread_max"] = float(np.nanmax(spread.values)) if len(sub) else np.nan
        s["network_spread_mean"] = float(np.nanmean(spread.values)) if len(sub) else np.nan
        # time of max spread
        if len(sub):
            idx = spread.idxmax()
            s["network_spread_tmax"] = str(idx)
            s["network_min_at_tmax"] = float(row_min.loc[idx])
            s["network_max_at_tmax"] = float(row_max.loc[idx])
        else:
            s["network_spread_tmax"] = ""
            s["network_min_at_tmax"] = np.nan
            s["network_max_at_tmax"] = np.nan
        return pd.Series(s, name=name)

    pre = stats_for_window("pre", t0, fw0)
    fw = stats_for_window("fireworks", fw0, fw1)
    post = stats_for_window("post", fw1, t1)
    return pd.DataFrame([pre, fw, post])


def read_public_csv(path: str, time_col: str, pm_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if time_col not in df.columns or pm_col not in df.columns:
        raise ValueError(f"Public CSV missing required columns: {time_col}, {pm_col}. Found: {df.columns.tolist()}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.rename(columns={time_col: "dt", pm_col: "pm25_public"})[["dt", "pm25_public"]]
    return df


def shade_fireworks(ax, fw0, fw1):
    """
    Shade the fireworks window and add label on the right side of the plot
    """
    ax.axvspan(fw0, fw1, alpha=0.15, color='gray', label='Fireworks', zorder=0)

def pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration (µg/m³) to AQI using EPA breakpoints.
    
    Args:
        pm25: PM2.5 concentration in µg/m³ (can be single value or array)
    
    Returns:
        AQI value (same type as input - single value or array)
    """
    import numpy as np
    
    # EPA AQI Breakpoints for PM2.5 (24-hour average, but commonly used for real-time)
    # [C_low, C_high, AQI_low, AQI_high, Category]
    breakpoints = [
        (0.0,    12.0,   0,   50,  "Good"),
        (12.1,   35.4,   51,  100, "Moderate"),
        (35.5,   55.4,   101, 150, "Unhealthy for Sensitive Groups"),
        (55.5,   150.4,  151, 200, "Unhealthy"),
        (150.5,  250.4,  201, 300, "Very Unhealthy"),
        (250.5,  350.4,  301, 400, "Hazardous"),
        (350.5,  500.4,  401, 500, "Hazardous"),
    ]
    
    # Handle both single values and arrays
    is_scalar = np.isscalar(pm25)
    pm25_array = np.atleast_1d(pm25)
    aqi_array = np.zeros_like(pm25_array)
    
    for i, conc in enumerate(pm25_array):
        if np.isnan(conc):
            aqi_array[i] = np.nan
            continue
            
        # Find appropriate breakpoint
        for c_low, c_high, aqi_low, aqi_high, category in breakpoints:
            if c_low <= conc <= c_high:
                # Linear interpolation formula
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low
                aqi_array[i] = round(aqi)
                break
        else:
            # Beyond highest breakpoint
            if conc > 500.4:
                aqi_array[i] = 500  # Cap at 500
            else:
                aqi_array[i] = np.nan
    
    return aqi_array[0] if is_scalar else aqi_array

def get_aqi_category(aqi):
    """
    Get AQI category and color for a given AQI value.
    
    Returns:
        tuple: (category_name, color_hex)
    """
    if aqi <= 50:
        return "Good", "#00E400"
    elif aqi <= 100:
        return "Moderate", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

def main():
    args = parse_args()

    tc = TimeConfig(
        event_date=args.event_date,
        start_time=args.start_time,
        end_time=args.end_time,
        fireworks_start=args.fireworks_start,
        fireworks_end=args.fireworks_end,
    )
    t0, t1, fw0, fw1 = build_time_bounds(tc)

    files = sorted(glob.glob(args.sensor_glob))

    if not files:
        raise SystemExit(f"No files match {args.sensor_glob}")

    # Read sensors
    sensor_frames = []
    label_idx = 1
    for fp in files:
        if os.path.basename(fp) == "Data_4.csv":
            label_idx += 1
            continue
        s = read_sensor_csv(fp, anchor_start=t0)
        s = s[(s["dt"] >= t0) & (s["dt"] <= t1)].copy()
        s["sensor"] = f"S{label_idx}"        
        sensor_frames.append(s)
        label_idx += 1

    long = pd.concat(sensor_frames, ignore_index=True)
    wide = long.pivot_table(index="dt", columns="sensor", values="pm25", aggfunc="mean").sort_index()
    # Convert each PM2.5 sample to AQI (same shape as wide)
    wide_aqi = wide.applymap(pm25_to_aqi)

    # Network summary signals
    network = pd.DataFrame(index=wide.index)
    # Network summary in PM2.5 (keep if you still want it)
    network = pd.DataFrame(index=wide.index)
    network["min"] = wide.min(axis=1, skipna=True)
    network["max"] = wide.max(axis=1, skipna=True)
    network["mean"] = wide.mean(axis=1, skipna=True)
    network["range"] = network["max"] - network["min"]

    # Network summary in AQI (THIS is what you should plot for AQI range)
    network_aqi = pd.DataFrame(index=wide_aqi.index)
    network_aqi["min"] = wide_aqi.min(axis=1, skipna=True)
    network_aqi["max"] = wide_aqi.max(axis=1, skipna=True)
    network_aqi["mean"] = wide_aqi.mean(axis=1, skipna=True)
    network_aqi["range"] = network_aqi["max"] - network_aqi["min"]

    # Save cleaned data
    out_clean = f"{args.out_prefix}_network_clean.csv"
    wide_out = wide.copy()
    wide_out["network_min"] = network["min"]
    wide_out["network_max"] = network["max"]
    wide_out["network_mean"] = network["mean"]
    wide_out["network_range"] = network["range"]
    wide_out.to_csv(out_clean, index_label="dt")

    # Window stats
    stats = compute_window_stats(wide, t0, t1, fw0, fw1)
    out_stats = f"{args.out_prefix}_window_stats.csv"
    stats.to_csv(out_stats, index=True)

    # Optional public overlay
    public_df = None
    if args.public_csv is not None:
        if args.public_time_col is None or args.public_pm25_col is None:
            raise SystemExit("--public_time_col and --public_pm25_col are required when using --public_csv")
        public_df = read_public_csv(args.public_csv, args.public_time_col, args.public_pm25_col)
        # Restrict to window for plotting clarity
        public_df = public_df[(public_df["dt"] >= t0) & (public_df["dt"] <= t1)].copy()

    # -------- Plot 1: all sensors (full y)
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in wide.columns:
        ax.plot(wide_aqi.index, wide_aqi[c].values, linewidth=0.8, label=c)
    ax.set_title("AQI (1 Hz) – All sensors")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("AQI")
    ax.set_xlim([t0, t1])
    ax.set_ylim([0, args.zoom_max])
    shade_fireworks(ax, fw0, fw1)
    ax.legend(ncol=7, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out1 = f"{args.out_prefix}_timeseries_all.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    # -------- Plot 2: zoomed y-limit
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in wide.columns:
        ax.plot(wide_aqi.index, wide_aqi[c].values, linewidth=0.8, label=c)
    ax.set_title(f"AQI (1 Hz) – Zoom (0–{args.zoom_max:g})")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("AQI")
    ax.set_xlim([t0, t1])
    ax.set_ylim([0, args.zoom_max])
    shade_fireworks(ax, fw0, fw1)
    ax.legend(ncol=7, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out2 = f"{args.out_prefix}_timeseries_zoom.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)

    # -------- Plot 3: spatial variability (range)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(network.index, network["range"].values, linewidth=1.0)
    ax.plot(network_aqi.index, network_aqi["range"].values, linewidth=1.0, label="AQI range (max–min)")
    ax.set_ylabel("AQI range")
    ax.set_xlabel("Time (local)")
    ax.set_xlim([t0, t1])
    ax.set_ylim([0, args.zoom_max])
    shade_fireworks(ax, fw0, fw1)
    ax.legend(ncol=7, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out3 = f"{args.out_prefix}_spatial_range.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)

    # -------- Plot 4 (optional): overlay public monitor
    if public_df is not None and len(public_df):
        # Resample both series so scales are comparable/readable
        res = args.public_resample
        wide_r = wide.resample(res).mean()
        net_r = pd.DataFrame({
            "mean": wide_r.mean(axis=1, skipna=True),
            "min": wide_r.min(axis=1, skipna=True),
            "max": wide_r.max(axis=1, skipna=True),
        })
        pub_r = public_df.set_index("dt")["pm25_public"].resample(res).mean()

        fig, ax = plt.subplots(figsize=(11, 5))
        # sensor network band
        ax.plot(net_r.index, net_r["mean"], linewidth=1.5, label=f"Sensor network mean ({res})")
        ax.fill_between(net_r.index, net_r["min"], net_r["max"], alpha=0.15, label="Sensor network min–max")
        # public monitor
        ax.plot(pub_r.index, pub_r.values, linewidth=1.5, label=f"Public monitor ({res})")

        ax.set_title("PM2.5 comparison: near-field sensor network vs public monitor")
        ax.set_xlabel("Time (local)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xlim([t0, t1])
        shade_fireworks(ax, fw0, fw1)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        out4 = f"{args.out_prefix}_compare_public.png"
        fig.savefig(out4, dpi=200)
        plt.close(fig)

    # Print key “X” numbers for your abstract
    # (max network spread during fireworks window)
    fw_net = network[(network.index >= fw0) & (network.index <= fw1)]
    if len(fw_net):
        tmax = fw_net["range"].idxmax()
        print("=== Fireworks-window max spatial spread ===")
        print(f"t_max_spread: {tmax}")
        print(f"min_at_tmax:  {fw_net.loc[tmax, 'min']:.3g} µg/m³")
        print(f"max_at_tmax:  {fw_net.loc[tmax, 'max']:.3g} µg/m³")
        print(f"spread:       {fw_net.loc[tmax, 'range']:.3g} µg/m³")
    else:
        print("No data in fireworks window after filtering. Check your times / event_date.")

    print("\nWrote:")
    print(" ", out_clean)
    print(" ", out_stats)
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
    if public_df is not None and len(public_df):
        print(" ", f"{args.out_prefix}_compare_public.png")


if __name__ == "__main__":
    main()
