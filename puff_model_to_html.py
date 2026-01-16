import math
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime

# ===== EDIT THESE =====
EVENT_DATE = "2025-07-04"
START_TIME = "20:00:00"
END_TIME   = "23:30:00"

# Fireworks source location (approx barge)
SRC_LAT, SRC_LON = 47.6320, -122.3380

# Your sensor coordinates
SENSOR_LATLON = {
    "S1": (47.63771, -122.32957),
    "S2": (47.64694, -122.32629),
    "S3": (47.64933, -122.33158),
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

WIND_CSV = "wind_speed_2025.csv"
OUT_HTML = "puff_model_heatmap_time.html"
# ======================

# Domain/grid
HALF_WIDTH_M = 2200
GRID_N = 70          # keep modest so it runs fast

# Puff model params (increase for more spread)
Q0 = 1.0
SIGMA0_M = 120.0     # <-- spread more
K_M2S = 120.0        # <-- spread more
TAU_DECAY_S = 7200.0 # <-- fade slower (or set None)

DT_MIN = 1           # release + update step (minutes)


def wind_to_uv(ws_mps, wd_from_deg):
    """Return (u_east, v_north) from speed and met 'from' direction."""
    if np.isnan(ws_mps) or np.isnan(wd_from_deg):
        return 0.0, 0.0
    wd_to = (wd_from_deg + 180.0) % 360.0
    theta = math.radians(wd_to)
    v_n = ws_mps * math.cos(theta)
    u_e = ws_mps * math.sin(theta)
    return u_e, v_n


def latlon_to_xy_m(lat, lon, lat0, lon0):
    R = 6371000.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R * dlon * math.cos(math.radians(lat0))
    y = R * dlat
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    R = 6371000.0
    lat0_rad = np.deg2rad(lat0)
    lat = lat0 + np.rad2deg(y / R)
    lon = lon0 + np.rad2deg(x / (R * np.cos(lat0_rad)))
    return lat, lon


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

    # collapse duplicate timestamps (important)
    met = met.groupby("dt", sort=True).mean(numeric_only=True)

    met = met.resample("1min").ffill()
    return met.loc[t0:t1]


def field_to_points(lat_grid, lon_grid, Z, stride=2, keep_frac=0.10):
    """
    Convert grid to heatmap points.
    - stride reduces number of points
    - keep_frac keeps only cells above keep_frac * max(Z) to avoid huge lists
    """
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


def main():
    t0 = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    t1 = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")
    times = pd.date_range(t0, t1, freq=f"{DT_MIN}min")

    wind = read_wind_local_1min(WIND_CSV, t0, t1)

    # local coords centered at source
    lat0, lon0 = SRC_LAT, SRC_LON
    src_x, src_y = latlon_to_xy_m(SRC_LAT, SRC_LON, lat0, lon0)

    # grid in meters
    xs = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    ys = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    lat_grid, lon_grid = xy_to_latlon(X, Y, lat0, lon0)

    # puff states
    puff_x = []
    puff_y = []
    puff_t = []

    frames = []
    labels = []

    dt_s = DT_MIN * 60.0

    for t in times:
        # release a new puff each step
        puff_x.append(src_x)
        puff_y.append(src_y)
        puff_t.append(t)

        # advect by this minute wind
        ws = float(wind.loc[t, "wind_speed"]) if t in wind.index else np.nan
        wd = float(wind.loc[t, "wind_dir"]) if t in wind.index else np.nan
        u, v = wind_to_uv(ws, wd)
        for k in range(len(puff_x)):
            puff_x[k] += u * dt_s
            puff_y[k] += v * dt_s

        # compute instantaneous field
        Z = np.zeros_like(X, dtype=float)
        for k in range(len(puff_x)):
            age_s = (t - puff_t[k]).total_seconds()
            if age_s < 0:
                continue
            sigma2 = SIGMA0_M**2 + 2.0 * K_M2S * age_s
            sigma2 = max(sigma2, 1.0)
            r2 = (X - puff_x[k])**2 + (Y - puff_y[k])**2
            base = (Q0 / (2.0 * math.pi * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
            if TAU_DECAY_S:
                base *= math.exp(-age_s / TAU_DECAY_S)
            Z += base

        # convert to points for folium heatmap
        pts = field_to_points(lat_grid, lon_grid, Z, stride=2, keep_frac=0.08)
        frames.append(pts)
        labels.append(t.strftime("%H:%M"))

    # build map
    center = (SRC_LAT, SRC_LON)
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # markers for sensors + source
    folium.Marker(center, popup="Fireworks source").add_to(m)
    for sid, (slat, slon) in SENSOR_LATLON.items():
        folium.CircleMarker((slat, slon), radius=6, popup=sid, fill=True).add_to(m)

    # IMPORTANT: do NOT set blur=30 etc. Keep defaults.
    HeatMapWithTime(
        frames,
        index=labels,
        radius=45,             # bigger radius = visually more spread
        min_opacity=0.35,
        max_opacity=0.95,
        use_local_extrema=True,
        auto_play=False
    ).add_to(m)

    m.save(OUT_HTML)
    print("Saved:", OUT_HTML)
    print("View with: python -m http.server 8000  then open http://localhost:8000/" + OUT_HTML)


if __name__ == "__main__":
    main()
