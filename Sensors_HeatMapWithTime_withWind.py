import glob, os, re, math
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime

# ====== EDIT THIS: your sensor coordinates ======
SENSOR_LATLON = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

EVENT_DATE = "2025-07-04"
START_TIME = "20:00:00"
END_TIME   = "23:30:00"

WIND_CSV = "wind_speed_2025.csv"   # the one you uploaded
SENSOR_GLOB = "Data_*.csv"

RESAMPLE = "1min"                  # heatmap frame step
N_PUFFS = 6                        # how many downwind points per sensor
FAN_DEG = 15                       # +/- degrees around downwind direction
FAN_N = 3                          # number of rays in the fan
SPREAD_FACTOR = 0.30               # fraction of 1-minute advection distance per puff
MAX_ADVECT_M = 900                 # cap max smear length (meters)

OUT_HTML = "lake_union_pm25_heatmap_wind.html"


def read_sensor_csv(path: str, anchor_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]

    pm_col = None
    for c in ["PM2.5_Env", "PM2.5_Std", "PM2.5"]:
        if c in df.columns:
            pm_col = c
            break
    if pm_col is None:
        raise ValueError(f"{path}: no PM2.5 column found")

    df["Counter"] = pd.to_numeric(df["Counter"], errors="coerce")
    df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
    df = df.dropna(subset=["Counter", pm_col]).copy()
    df["Counter"] = df["Counter"].astype(int)
    df["dt"] = anchor_start + pd.to_timedelta(df["Counter"], unit="s")
    df = df.rename(columns={pm_col: "pm25"})
    return df[["dt", "pm25"]].sort_values("dt")


def read_wind_local_1min(path: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """
    Your wind_speed_2025.csv has columns like DATE (UTC), wind_speed (m/s), wind_direction (deg).
    wind_direction sometimes is 999 meaning missing/variable.
    """
    w = pd.read_csv(path, low_memory=False)
    if "DATE" not in w.columns:
        raise ValueError("wind CSV missing DATE column")

    w["dt"] = pd.to_datetime(w["DATE"], utc=True, errors="coerce") \
                .dt.tz_convert("America/Los_Angeles") \
                .dt.tz_localize(None)

    w["wind_speed_mps"] = pd.to_numeric(w.get("wind_speed"), errors="coerce")
    w["wind_dir_from_deg"] = pd.to_numeric(w.get("wind_direction"), errors="coerce")
    w.loc[w["wind_dir_from_deg"] == 999, "wind_dir_from_deg"] = np.nan

    w = w.dropna(subset=["dt"]).sort_values("dt")
    w = w[(w["dt"] >= t0.floor("h") - pd.Timedelta(hours=2)) & (w["dt"] <= t1.ceil("h") + pd.Timedelta(hours=2))]

    w = w.set_index("dt")[["wind_speed_mps", "wind_dir_from_deg"]].resample(RESAMPLE).ffill()
    w = w[(w.index >= t0) & (w.index <= t1)]
    return w


def meters_to_latlon(d_north_m: float, d_east_m: float, lat_deg: float):
    # Simple local conversion (good enough for < ~1 km)
    lat_rad = math.radians(lat_deg)
    dlat = d_north_m / 111320.0
    dlon = d_east_m / (111320.0 * math.cos(lat_rad))
    return dlat, dlon


def smear_downwind_points(lat, lon, value, ws_mps, wd_from_deg,
                          n_puffs=N_PUFFS, fan_deg=FAN_DEG, fan_n=FAN_N,
                          spread_factor=SPREAD_FACTOR, max_advect_m=MAX_ADVECT_M):
    """
    Create a small downwind 'plume' by adding multiple points along the wind direction.
    - value is PM2.5 at the sensor (µg/m³)
    - ws_mps controls how far the plume stretches
    - wd_from_deg is meteorological "from" direction; downwind is +180 deg
    Returns list of [lat, lon, weight] points.
    """
    if value is None or np.isnan(value):
        return []

    if ws_mps is None or np.isnan(ws_mps) or ws_mps < 0:
        ws_mps = 0.0

    # If wind direction missing, just return the point itself (no directional smear)
    if wd_from_deg is None or np.isnan(wd_from_deg):
        return [[lat, lon, float(value)]]

    # Convert to downwind "to" direction
    wd_to = (float(wd_from_deg) + 180.0) % 360.0

    # 1-minute advection distance in meters
    advect_1min = ws_mps * 60.0
    step = min(advect_1min * spread_factor, max_advect_m / max(1, n_puffs))
    step = max(step, 10.0)  # minimum step so it visibly spreads

    # weights along distance: exponential decay, normalized to conserve total "mass"
    idx = np.arange(n_puffs)
    wdist = np.exp(-idx / 2.0)
    wdist = wdist / wdist.sum()

    # fan angles around downwind direction
    if fan_n <= 1:
        fan_angles = [0.0]
    else:
        fan_angles = np.linspace(-fan_deg, fan_deg, fan_n)

    pts = []
    for i in range(n_puffs):
        dist = i * step
        for da in fan_angles:
            theta = math.radians((wd_to + da) % 360.0)  # bearing clockwise from north
            d_north = dist * math.cos(theta)
            d_east  = dist * math.sin(theta)

            dlat, dlon = meters_to_latlon(d_north, d_east, lat)
            wgt = float(value) * float(wdist[i]) / len(fan_angles)
            pts.append([lat + dlat, lon + dlon, wgt])

    # also include the sensor point itself (strongest)
    pts.append([lat, lon, float(value)])
    return pts


def main():
    t0 = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    t1 = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")

    # ---- Load sensors -> 1-min mean ----
    series = {}
    for fp in sorted(glob.glob(SENSOR_GLOB)):
        m = re.search(r"Data_(\d+)\.csv$", os.path.basename(fp))
        if not m:
            continue
        snum = int(m.group(1))
        if snum == 4:
            continue
        sid = f"S{snum}"
        if sid not in SENSOR_LATLON:
            continue

        d = read_sensor_csv(fp, anchor_start=t0)
        d = d[(d["dt"] >= t0) & (d["dt"] <= t1)]
        s = d.set_index("dt")["pm25"].resample(RESAMPLE).mean()
        series[sid] = s

    wide = pd.concat(series, axis=1).sort_index()

    # ---- Load wind -> 1-min ----
    wind = read_wind_local_1min(WIND_CSV, t0, t1)

    # ---- Build frames with downwind smear ----
    frames = []
    labels = []
    for ts, row in wide.iterrows():
        pts = []
        ws = float(wind.loc[ts, "wind_speed_mps"]) if ts in wind.index else np.nan
        wd = float(wind.loc[ts, "wind_dir_from_deg"]) if ts in wind.index else np.nan

        for sid, val in row.items():
            if pd.isna(val):
                continue
            lat, lon = SENSOR_LATLON[sid]
            pts.extend(smear_downwind_points(lat, lon, float(val), ws, wd))

        frames.append(pts)
        labels.append(ts.strftime("%H:%M"))

    # ---- Map ----
    center = np.mean([v[0] for v in SENSOR_LATLON.values()]), np.mean([v[1] for v in SENSOR_LATLON.values()])
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # markers so you know you're in the right place
    for sid, (lat, lon) in SENSOR_LATLON.items():
        folium.CircleMarker((lat, lon), radius=6, popup=sid, fill=True).add_to(m)
    m.fit_bounds(list(SENSOR_LATLON.values()))

    HeatMapWithTime(
        frames,
        index=labels,
        radius=30,
        min_opacity=0.35,
        max_opacity=0.95,
        use_local_extrema=True,   # makes colors use the full range each time step
        auto_play=False
    ).add_to(m)

    m.save(OUT_HTML)
    print("Saved:", OUT_HTML)


if __name__ == "__main__":
    main()
