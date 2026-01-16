#!/usr/bin/env python3
"""
Fireworks AQI analysis (3 plots)

Plots:
1) Monitoring stations PM2.5 AQI time series (hourly) in the event window
2) Monitoring stations spatial variability (range = max–min) of PM2.5 AQI
3) Sensor network AQI range (computed from sensor PM2.5 hourly means) vs monitor AQI range

Notes:
- AirQualityWA "AQI Report.csv" files: first line is station name(s) -> skipped
- Low-cost sensors are 1 Hz PM2.5; we compute hourly mean PM2.5 per sensor, then convert to AQI
- AQI conversion below uses the standard PM2.5 AQI breakpoints (24-hr AQI formula) applied to hourly means
  (this is an estimate; it won’t match “NowCast AQI” perfectly, but it’s consistent for comparison).

Run example (PowerShell):
python plot_stations_fireworks_pm25.py --event_date 2025-07-04 --sensor_glob "Data_*.csv" --skip_sensor 4 --monitor "Seattle-Beacon Hill AQI Report.csv" "Seattle-10th & Weller AQI Report.csv" "Seattle-Duwamish AQI Report.csv" "Seattle-College Way N AQI Report.csv" "Seattle-Linden Ave N AQI Report.csv" "Seattle-NE 127th St AQI Report.csv" --start_time 20:00 --end_time 23:30 --fireworks_start 22:20 --fireworks_end 22:40 --out_dir .

"""

from __future__ import annotations
import argparse, os, re, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--event_date", required=True, help="YYYY-MM-DD (local)")
    p.add_argument("--start_time", default="20:00", help="HH:MM local (default 20:00)")
    p.add_argument("--end_time", default="23:30", help="HH:MM local (default 23:30)")
    p.add_argument("--fireworks_start", default="22:20", help="HH:MM local (default 22:20)")
    p.add_argument("--fireworks_end", default="22:40", help="HH:MM local (default 22:40)")

    p.add_argument("--sensor_glob", default="Data_*.csv", help='Sensor files glob (default "Data_*.csv")')
    p.add_argument("--skip_sensor", type=int, default=4, help="Sensor number to skip (default 4)")

    # Allow 3+ stations
    p.add_argument("--monitor", nargs="+", required=True,
                   help='AirQualityWA "AQI Report.csv" files (3 or more)')

    # Prefer PM2.5 AQI if present; fallback to Overall AQI
    p.add_argument("--monitor_aqi_col", default="PM2.5 AQI",
                   help='Column name to use from AQI report (default "PM2.5 AQI"). '
                        'Will fall back to "Overall AQI" if not found.')

    p.add_argument("--out_dir", default=".", help="Output directory")
    return p.parse_args()


def shade_fireworks(ax, fw0: pd.Timestamp, fw1: pd.Timestamp):
    # Grey range legend item on every plot
    ax.axvspan(fw0, fw1, alpha=0.20, color="0.6", label="Fireworks window")


def infer_site_name(path: str) -> str:
    base = os.path.basename(path)
    # "Seattle-NE 127th St AQI Report.csv" -> "NE127thSt"
    m = re.search(r"Seattle-(.*)\s+AQI Report", base)
    if m:
        return m.group(1).strip().replace(" ", "").replace("&", "")
    return os.path.splitext(base)[0].replace(" ", "").replace("&", "")


def read_airqualitywa_aqi_report(csv_path: str, preferred_col: str) -> pd.DataFrame:
    """
    AirQualityWA "AQI Report.csv":
      line 1: station name(s) -> skip
      line 2: header (Date/Time, Overall AQI, Description, Dominant Pollutant, PM2.5 AQI, ...)
    """
    df = pd.read_csv(csv_path, skiprows=1)
    # Drop unnamed trailing columns (common because the CSV ends with a comma)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Date/Time" not in df.columns:
        raise ValueError(f"{os.path.basename(csv_path)}: missing 'Date/Time'. Columns: {df.columns.tolist()}")

    # Pick AQI column
    aqi_col = None
    if preferred_col in df.columns:
        aqi_col = preferred_col
    elif "Overall AQI" in df.columns:
        aqi_col = "Overall AQI"
    else:
        # last resort: any column containing 'AQI'
        for c in df.columns:
            if "AQI" in c:
                aqi_col = c
                break
    if aqi_col is None:
        raise ValueError(f"{os.path.basename(csv_path)}: no AQI column found. Columns: {df.columns.tolist()}")

    df["dt"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df["aqi"] = pd.to_numeric(df[aqi_col], errors="coerce")
    out = df[["dt", "aqi"]].dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def read_lowcost_sensor_csv(csv_path: str, anchor_start: pd.Timestamp) -> pd.DataFrame:
    """
    Low-cost sensor CSV:
      - First line is device id -> skip
      - Counter is seconds from power-on
      - dt = anchor_start + Counter seconds
    """
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]

    if "Counter" not in df.columns:
        raise ValueError(f"{os.path.basename(csv_path)}: missing 'Counter' column.")

    pm_col = None
    for c in ["PM2.5_Env", "PM2.5_Std", "PM2.5"]:
        if c in df.columns:
            pm_col = c
            break
    if pm_col is None:
        raise ValueError(f"{os.path.basename(csv_path)}: missing PM2.5 column (PM2.5_Env/PM2.5_Std/PM2.5).")

    df["Counter"] = pd.to_numeric(df["Counter"], errors="coerce")
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
    df = df.dropna(subset=["Counter"]).copy()
    df["Counter"] = df["Counter"].astype(int)
    df["dt"] = anchor_start + pd.to_timedelta(df["Counter"], unit="s")
    df = df.rename(columns={pm_col: "pm25"})
    return df[["dt", "pm25"]].sort_values("dt")


def pm25_to_aqi(pm: float) -> float:
    """
    Convert PM2.5 (µg/m³) to AQI using standard EPA breakpoints (24-hr AQI formula).
    Returns float; you can round later.
    """
    if pm is None or (isinstance(pm, float) and np.isnan(pm)):
        return np.nan

    # Breakpoints: (C_low, C_high, I_low, I_high)
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    # Clamp to max breakpoint
    if pm > 500.4:
        pm = 500.4
    if pm < 0:
        pm = 0.0

    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= pm <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (pm - c_lo) + i_lo

    return np.nan


def main():
    args = parse_args()

    if len(args.monitor) < 3:
        raise SystemExit("Please provide at least 3 monitor AQI CSV files.")

    t0 = pd.Timestamp(f"{args.event_date} {args.start_time}:00")
    t1 = pd.Timestamp(f"{args.event_date} {args.end_time}:00")
    fw0 = pd.Timestamp(f"{args.event_date} {args.fireworks_start}:00")
    fw1 = pd.Timestamp(f"{args.event_date} {args.fireworks_end}:00")

    if not (t0 <= fw0 <= fw1 <= t1):
        raise SystemExit("Time bounds must satisfy start <= fireworks_start <= fireworks_end <= end")

    # Let monitors include the next hour tick (e.g., 23:30 -> 00:00 next day) so 12:00 AM can show
    t1_mon = t1.ceil("h")

    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------
    # Load + FILTER monitors (AQI)
    # ----------------------------
    mon_series = []
    for pth in args.monitor:
        name = infer_site_name(pth)
        d = read_airqualitywa_aqi_report(pth, preferred_col=args.monitor_aqi_col)

        d = d[(d["dt"] >= t0) & (d["dt"] <= t1_mon)].copy()
        d = d.dropna(subset=["aqi"])

        if d.empty:
            print(f"[WARN] Monitor {name} has no AQI values in {t0}–{t1_mon}.")
            continue

        mon_series.append(d.set_index("dt")["aqi"].rename(name))

    if not mon_series:
        raise SystemExit("No monitor AQI data in the requested window. Check files/times.")

    mon_wide = pd.concat(mon_series, axis=1).sort_index()
    mon_range = mon_wide.max(axis=1, skipna=True) - mon_wide.min(axis=1, skipna=True)

    # ----------------------------
    # Load + FILTER sensors -> compute hourly mean PM2.5 per sensor -> AQI -> range
    # ----------------------------
    sensor_files = sorted(glob.glob(args.sensor_glob))
    if not sensor_files:
        raise SystemExit(f"No sensor files match {args.sensor_glob}")

    sensor_long = []
    for fp in sensor_files:
        m = re.search(r"Data_(\d+)\.csv$", os.path.basename(fp))
        if not m:
            continue
        snum = int(m.group(1))
        if snum == args.skip_sensor:
            continue

        d = read_lowcost_sensor_csv(fp, anchor_start=t0)
        d = d[(d["dt"] >= t0) & (d["dt"] <= t1)].copy()
        if d.empty:
            continue
        d["sensor"] = f"S{snum}"
        sensor_long.append(d)

    if not sensor_long:
        raise SystemExit("No sensor data in the requested window. Check sensor files/times.")

    sensor_long = pd.concat(sensor_long, ignore_index=True)
    sensor_long = sensor_long.set_index("dt")

    # Hourly mean PM2.5 per sensor (clock-hour bins)
    pm_hourly = (sensor_long
                 .groupby("sensor")["pm25"]
                 .resample("h")
                 .mean()
                 .unstack("sensor")
                 .sort_index())

    # Convert each hourly mean PM2.5 to AQI
    aqi_hourly = pm_hourly.applymap(pm25_to_aqi).round(0)  # round to integer-like AQI

    # Sensor network AQI range (max-min across sensors per hour)
    sensor_range_aqi_hourly = aqi_hourly.max(axis=1, skipna=True) - aqi_hourly.min(axis=1, skipna=True)

    # ----------------------------
    # Plot 1: monitors AQI (event window)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in mon_wide.columns:
        ax.plot(mon_wide.index, mon_wide[col].values, marker="o", linewidth=1.4, label=col)
    shade_fireworks(ax, fw0, fw1)
    ax.set_xlim([t0, t1_mon])
    ax.set_title("Regulatory monitors PM2.5 AQI (hourly) – event window")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("AQI")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot1_monitors_aqi_event_window.png"), dpi=200)
    plt.close(fig)

    # ----------------------------
    # Plot 2: monitors AQI spatial variability (range)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mon_range.index, mon_range.values, marker="o", linewidth=1.4, label="Monitor AQI range (max–min)")
    shade_fireworks(ax, fw0, fw1)
    ax.set_xlim([t0, t1_mon])
    ax.set_title("Regulatory monitors spatial variability (max–min) – AQI")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("AQI range")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot2_monitors_aqi_spatial_range.png"), dpi=200)
    plt.close(fig)

    # ----------------------------
    # Plot 3: sensor AQI range (hourly) vs monitor AQI range (hourly)
    # ----------------------------
    common_x = mon_range.index
    y_sensor = sensor_range_aqi_hourly.reindex(common_x)
    y_mon = mon_range.reindex(common_x)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(common_x, y_sensor.values, marker="o", linewidth=1.4,
            label=f"Sensor network AQI range (hourly; computed from PM2.5; skip S{args.skip_sensor})")
    ax.plot(common_x, y_mon.values, marker="o", linewidth=1.4,
            label="Regulatory monitor AQI range (hourly)")
    shade_fireworks(ax, fw0, fw1)
    ax.set_xlim([t0, t1_mon])
    ax.set_title("Spatial variability (range) comparison – AQI (sensors vs monitors)")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("AQI range")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plot3_range_compare_sensors_vs_monitors_aqi.png"), dpi=200)
    plt.close(fig)

    print("Wrote 3 plots into:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
