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
    "S4": (47.63629, -122.33998),
    "S5": (47.62985, -122.33937),
    "S6": (47.62674, -122.33528),
}

OUT_DIR = "calibrated_centered_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Heatmap domain/grid (meters around source)
HALF_WIDTH_M = 3500
GRID_N = 160
DT_MIN = 1  # 1-minute model step
WIND_STRETCH_L = 50.0   # meters, 风对“下风向扩散”的增强尺度(越大越拉长)
K_PERP_SCALE   = 1.0    # 横风扩散倍率(一般=1)

# Heatmap point conversion
GRID_STRIDE = 1
KEEP_FRAC = 0.02
HEAT_RADIUS = 70

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

def wind_from_to_unit_vectors(wd_from_deg):
    """
    meteorological 'from' deg -> unit vectors (downwind, crosswind) in (east,north)
    returns: (epar_x, epar_y, eperp_x, eperp_y)
    """
    if np.isnan(wd_from_deg):
        # no wind direction -> default axes
        return 1.0, 0.0, 0.0, 1.0

    wd_to = (wd_from_deg + 180.0) % 360.0
    theta = math.radians(wd_to)

    # downwind unit vector in EN coords
    epar_x = math.sin(theta)   # east component
    epar_y = math.cos(theta)   # north component

    # crosswind unit vector (rotate +90 deg)
    eperp_x = math.cos(theta)
    eperp_y = -math.sin(theta)
    return epar_x, epar_y, eperp_x, eperp_y


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
        # if snum == 4:
        #     continue
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
def predict_unitQ_at_sensors(times, ws_mps, wind_dir_from, src_xy, sensor_xy,
                             sigma0_m, K_m2s, tau_s=None, release_mask=None, dt_s=60.0,
                             wind_stretch_L=60.0,  # m, controls downwind enhancement
                             U_cap=4.0,            # m/s cap to avoid extreme stretching
                             K_perp_scale=1.0):
    """
    Fixed-center + wind-shaped + per-puff locked wind (RECOMMENDED):
    - Puff centers NEVER move from src_xy
    - Each puff locks (wind speed, wind direction) at release time
    - Downwind diffusivity: K_par = K + L * U_eff
    - Crosswind diffusivity: K_perp = K * K_perp_scale
    - Optional decay: exp(-age/tau)
    """
    T = len(times)
    Ns = sensor_xy.shape[0]
    if release_mask is None:
        release_mask = np.ones(T, dtype=bool)

    # constant dx,dy from source to sensors
    dx = sensor_xy[:, 0] - src_xy[0]  # east meters
    dy = sensor_xy[:, 1] - src_xy[1]  # north meters

    # store puff attributes at release time
    puff_release = []
    puff_sin = []
    puff_cos = []
    puff_U = []

    G = np.zeros((T, Ns), dtype=float)

    for t_idx in range(T):
        # release puff
        if release_mask[t_idx]:
            wd_from = wind_dir_from[t_idx]
            U = ws_mps[t_idx]

            if np.isfinite(wd_from) and np.isfinite(U) and U > 0:
                wd_to = (wd_from + 180.0) % 360.0
                theta = math.radians(wd_to)
                puff_sin.append(math.sin(theta))
                puff_cos.append(math.cos(theta))
                puff_U.append(float(min(U, U_cap)))
            else:
                # calm/unknown: default axis
                puff_sin.append(0.0)
                puff_cos.append(1.0)
                puff_U.append(0.0)

            puff_release.append(t_idx)

        # sum all existing puffs
        for k in range(len(puff_release)):
            age_s = (t_idx - puff_release[k]) * dt_s
            if age_s < 0:
                continue

            sin_t = puff_sin[k]
            cos_t = puff_cos[k]
            Ueff  = puff_U[k]

            # rotate sensor offsets into wind coords for THIS puff
            # x_par = projection onto downwind (east,north) = (sin,cos)
            xpar = dx * sin_t + dy * cos_t
            # x_perp = crosswind axis
            xper = dx * cos_t - dy * sin_t

            K_par  = K_m2s + wind_stretch_L * Ueff
            K_perp = K_m2s * K_perp_scale

            sig_par2  = sigma0_m**2 + 2.0 * K_par  * age_s
            sig_perp2 = sigma0_m**2 + 2.0 * K_perp * age_s
            sig_par2  = max(sig_par2, 1.0)
            sig_perp2 = max(sig_perp2, 1.0)

            norm = 1.0 / (2.0 * math.pi * math.sqrt(sig_par2 * sig_perp2))
            expo = np.exp(-0.5 * (xpar*xpar/sig_par2 + xper*xper/sig_perp2))
            base = norm * expo

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

def grid_search(y, times, ws_fit, wind_dir_from, src_xy, sensor_xy, release_mask, w=None):
    # Coarse but robust parameter grid
    sigma0_list = [30, 60, 100, 150, 220]          # meters
    K_list      = [10, 30, 80, 150, 250]           # m^2/s
    tau_list    = [900, 1200, 2400, 7200]         # seconds

    best = {"rmse": 1e99}
    for sigma0 in sigma0_list:
        for K in K_list:
            for tau in tau_list:
                G = predict_unitQ_at_sensors(
                    times,
                    ws_fit, wind_dir_from,          # ✅ NEW: wind speed + met "FROM" direction
                    src_xy, sensor_xy,
                    sigma0_m=sigma0, K_m2s=K, tau_s=tau,
                    release_mask=release_mask, dt_s=60.0,
                    wind_stretch_L=60.0, U_cap=5.0, K_perp_scale=1.0
                )

                Q, rmse = fit_Q_and_rmse(y, G, w=w)
                if rmse < best["rmse"]:
                    best = {"sigma0": sigma0, "K": K, "tau": tau, "Q": Q, "rmse": rmse}
    return best

# =======================
# Model field for HTML
# =======================
def field_to_points(lat_grid, lon_grid, Z_norm,
                    stride=6, max_points=1200, wmin=0.02):
    """
    Convert 2D normalized field (0..1) -> [lat, lon, weight] points.
    This aggressively reduces points per frame so HeatMapWithTime can render.

    stride: subsample every stride grid cells
    max_points: keep only top-K points per frame
    wmin: drop weak weights (in 0..1)
    """
    lat = lat_grid[::stride, ::stride].ravel()
    lon = lon_grid[::stride, ::stride].ravel()
    w   = Z_norm[::stride, ::stride].ravel()

    m = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(w) & (w >= wmin)
    lat, lon, w = lat[m], lon[m], w[m]

    if w.size == 0:
        return []

    # keep top-K strongest points
    if w.size > max_points:
        idx = np.argpartition(w, -max_points)[-max_points:]
        lat, lon, w = lat[idx], lon[idx], w[idx]

    return [[float(a), float(b), float(c)] for a, b, c in zip(lat, lon, w)]


def precompute_idw_weights(X, Y, sensor_xy, power=2.0, eps_m=5.0):
    """
    Precompute inverse-distance weights from sensors to every grid cell.
    X,Y: 2D grid in meters
    sensor_xy: (Ns,2) in meters [east,north]
    Returns W: (M,Ns) where M = X.size
    """
    gx = X.ravel()[:, None]
    gy = Y.ravel()[:, None]
    sx = sensor_xy[:, 0][None, :]
    sy = sensor_xy[:, 1][None, :]

    D = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2) + eps_m
    W = 1.0 / (D ** power)
    return W

def idw_field_from_weights(W, values, out_shape):
    """
    W: (M,Ns) weights
    values: (Ns,) sensor values (may contain NaN)
    out_shape: X.shape
    """
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.zeros(out_shape, dtype=float)

    Wm = W[:, mask]
    vm = values[mask]
    denom = Wm.sum(axis=1)
    field = (Wm @ vm) / np.maximum(denom, 1e-12)
    return field.reshape(out_shape)


def simulate_frames(times, ws_mps, wind_dir_from, src_xy, X, Y, lat_grid, lon_grid,
                    wide_full, sensors, W_idw, baseline_freeze_vals, fw0,
                    *, sigma0, K, tau, Q, release_mask,
                    dt_s=60.0, wind_stretch_L=60.0, U_cap=5.0, K_perp_scale=1.0):
    """
    Fixed-center + wind-shaped + per-puff locked wind field for HTML frames.
    """
    frames = []
    labels = []

    # grid offsets from source (constant)
    dX = X - src_xy[0]
    dY = Y - src_xy[1]

    puff_release = []
    puff_sin = []
    puff_cos = []
    puff_U = []

    for t_idx, t in enumerate(times):
        # release puff
        if release_mask[t_idx]:
            wd_from = wind_dir_from[t_idx]
            U = ws_mps[t_idx]
            if np.isfinite(wd_from) and np.isfinite(U) and U > 0:
                wd_to = (wd_from + 180.0) % 360.0
                theta = math.radians(wd_to)
                puff_sin.append(math.sin(theta))
                puff_cos.append(math.cos(theta))
                puff_U.append(float(min(U, U_cap)))
            else:
                puff_sin.append(0.0)
                puff_cos.append(1.0)
                puff_U.append(0.0)
            puff_release.append(t_idx)

        Z = np.zeros_like(X, dtype=float)

        # sum all puffs
        for k in range(len(puff_release)):
            age_s = (t_idx - puff_release[k]) * dt_s
            if age_s < 0:
                continue

            sin_t = puff_sin[k]
            cos_t = puff_cos[k]
            Ueff  = puff_U[k]

            # rotate grid offsets into this puff's wind coordinates
            Xpar = dX * sin_t + dY * cos_t
            Xper = dX * cos_t - dY * sin_t

            K_par  = K + wind_stretch_L * Ueff
            K_perp = K * K_perp_scale

            sig_par2  = sigma0**2 + 2.0 * K_par  * age_s
            sig_perp2 = sigma0**2 + 2.0 * K_perp * age_s
            sig_par2  = max(sig_par2, 1.0)
            sig_perp2 = max(sig_perp2, 1.0)

            norm = Q / (2.0 * math.pi * math.sqrt(sig_par2 * sig_perp2))
            expo = np.exp(-0.5 * (Xpar*Xpar/sig_par2 + Xper*Xper/sig_perp2))
            base = norm * expo

            if tau is not None and tau > 0:
                base *= math.exp(-age_s / tau)

            Z += base
            
        t_now = pd.Timestamp(times[t_idx])  # 当前帧时间

        # dynamic baseline: pre-fireworks use observed sensors; after fireworks freeze
        if t_now < fw0:
            baseline_vals_t = wide_full.loc[t_now, sensors].to_numpy(dtype=float)
        else:
            baseline_vals_t = baseline_freeze_vals

        baseline_grid_t = idw_field_from_weights(W_idw, baseline_vals_t, X.shape)

        Z_total = baseline_grid_t + Z

        frames.append(field_to_points(lat_grid, lon_grid, Z_total, stride=6, max_points=1200, wmin=0.02))
        labels.append(pd.Timestamp(times[t_idx]).strftime("%H:%M"))


    return frames, labels


# =======================
# Main
# =======================
def main():
    # ---------------- Time bounds ----------------
    t0 = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    t1 = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")
    fw0 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_START}")
    fw1 = pd.Timestamp(f"{EVENT_DATE} {FIREWORKS_END}")
    t_baseline_end = pd.Timestamp(f"{EVENT_DATE} {BASELINE_END}")
    t_fit_end      = pd.Timestamp(f"{EVENT_DATE} {FIT_END}")

    # ---------------- 1) Load sensors/wind (1-min) ----------------
    wide = load_all_sensors_1min(t0, t1)
    wind = read_wind_local_1min(WIND_CSV, t0, t1)

    # Align by intersection
    idx = wide.index.intersection(wind.index)
    wide = wide.loc[idx]
    wind = wind.loc[idx]

    # ---------------- 2) Fitting window ----------------
    fit_idx = wide.index[(wide.index >= t0) & (wide.index <= t_fit_end)]
    wide_fit = wide.loc[fit_idx]
    wind_fit = wind.loc[fit_idx]

    # ---------------- 3) Baseline subtraction (for calibration only) ----------------
    base_idx = wide.index[(wide.index >= t0) & (wide.index <= t_baseline_end)]
    baselines = wide.loc[base_idx].median(axis=0, skipna=True)

    y = (wide_fit - baselines).fillna(0.0).clip(lower=0.0)

    # ---------------- 4) Local meter coordinates ----------------
    lat0, lon0 = SRC_LAT, SRC_LON
    src_xy = latlon_to_xy_m(SRC_LAT, SRC_LON, lat0, lon0)

    sensors = list(wide_fit.columns)
    sensor_xy = np.array([
        latlon_to_xy_m(SENSOR_LATLON[s][0], SENSOR_LATLON[s][1], lat0, lon0)
        for s in sensors
    ], dtype=float)

    # ---------------- 5) Wind arrays (fit) ----------------
    ws_fit = wind_fit["wind_speed"].to_numpy(dtype=float)      # m/s
    wd_from_fit = wind_fit["wind_dir"].to_numpy(dtype=float)   # deg FROM

    times = wide_fit.index.to_pydatetime()

    # release only during fireworks (fit window)
    release_mask = np.array([(t >= fw0) and (t < fw1) for t in wide_fit.index], dtype=bool)

    # ---------------- 6) Grid search calibration ----------------
    best = grid_search(y.values, times, ws_fit, wd_from_fit, src_xy, sensor_xy, release_mask)
    print("=== BEST PARAMS (from grid search) ===")
    print(best)

    # ---------------- 7) Predicted vs observed at sensors (plot) ----------------
    G_best = predict_unitQ_at_sensors(
        times,
        ws_fit,
        wd_from_fit,
        src_xy, sensor_xy,
        sigma0_m=best["sigma0"], K_m2s=best["K"], tau_s=best["tau"],
        release_mask=release_mask, dt_s=60.0,
        wind_stretch_L=60.0, U_cap=5.0, K_perp_scale=1.0
    )
    pred_enh = best["Q"] * G_best
    pred = pred_enh + baselines[sensors].values  # add baseline back

    pred_df = pd.DataFrame(pred, index=wide_fit.index, columns=sensors)
    obs_df = wide_fit.copy()

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

    # ---------------- 8) Build grid (meters) ----------------
    xs = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    ys = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    lat_grid, lon_grid = xy_to_latlon(X, Y, lat0, lon0)

    # Precompute IDW weights (used inside simulate_frames for dynamic baseline)
    W_idw = precompute_idw_weights(X, Y, sensor_xy, power=2.0, eps_m=5.0)

    # ---------------- 9) Full window time axis + full wind + full sensor ----------------
    full_times = pd.date_range(t0, t1, freq=f"{DT_MIN}min")

    wind_full = wind.reindex(full_times).interpolate(limit=3).ffill().bfill()
    ws_full = wind_full["wind_speed"].to_numpy(dtype=float)
    wd_from_full = wind_full["wind_dir"].to_numpy(dtype=float)

    wide_full = wide.reindex(full_times).interpolate(limit=3).ffill().bfill()
    wide_full.index = pd.DatetimeIndex(wide_full.index).floor("min")  # 防止索引不齐

    # Freeze baseline at 1 minute before fireworks
    t_freeze = (fw0 - pd.Timedelta(minutes=1)).floor("min")
    if t_freeze not in wide_full.index:
        t_freeze = wide_full.index[wide_full.index < fw0][-1]
    baseline_freeze_vals = wide_full.loc[t_freeze, sensors].to_numpy(dtype=float)

    release_full = np.array([(t >= fw0) and (t < fw1) for t in full_times], dtype=bool)

    # ---------------- 10) Simulate heatmap frames ----------------
    frames, labels = simulate_frames(
        full_times.to_pydatetime(),
        ws_full, wd_from_full,
        src_xy, X, Y, lat_grid, lon_grid,
        wide_full, sensors, W_idw, baseline_freeze_vals, fw0,
        sigma0=best["sigma0"], K=best["K"], tau=best["tau"], Q=best["Q"],
        release_mask=release_full
    )

    # ---------------- 11) Normalize + SLIM DOWN frames (关键：减少HTML点数) ----------------
    # 11.1 choose a robust CMAX by percentile (prevents "red carpet" & keeps contrast)
    raw = np.array([p[2] for fr in frames for p in fr], dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size:
        CMAX = float(np.nanpercentile(raw, 98))
        CMAX = max(CMAX, 20.0)
    else:
        CMAX = 150.0
    print("DEBUG: CMAX used =", CMAX)

    # 11.2 scale to 0..1
    frames = [
        [[lat, lon, max(0.0, min(w, CMAX)) / CMAX] for (lat, lon, w) in fr]
        for fr in frames
    ]

    # 11.3 per-frame filtering: drop weak points + keep Top-K only
    MAX_POINTS_PER_FRAME = 1200   # 推荐 500~1500
    WMIN = 0.02                   # 推荐 0.02~0.06（越大越不铺满）

    frames_slim = []
    for fr in frames:
        fr2 = [pt for pt in fr if np.isfinite(pt[2]) and pt[2] >= WMIN]
        if len(fr2) > MAX_POINTS_PER_FRAME:
            fr2 = sorted(fr2, key=lambda x: x[2])[-MAX_POINTS_PER_FRAME:]
        frames_slim.append(fr2)
    frames = frames_slim

    print("DEBUG: frames =", len(frames))
    print("DEBUG: points in first frame =", len(frames[0]) if len(frames) else None)
    print("Total points:", sum(len(fr) for fr in frames))

    # ---------------- 12) Build folium map + HeatMapWithTime ----------------
    m = folium.Map(location=(SRC_LAT, SRC_LON), zoom_start=13, tiles="OpenStreetMap")

    folium.Marker((SRC_LAT, SRC_LON), popup="Fireworks Source").add_to(m)
    for s, (slat, slon) in SENSOR_LATLON.items():
        folium.CircleMarker((slat, slon), radius=6, popup=s, fill=True).add_to(m)

    HeatMapWithTime(
        frames,
        index=labels,
        radius=26,
        blur=20,
        min_opacity=0.15,
        max_opacity=0.95,
        use_local_extrema=False,
        auto_play=False
    ).add_to(m)

    out_html = os.path.join(OUT_DIR, "calibrated_puff_plume_time.html")
    m.save(out_html)
    print("Saved:", out_html)

    print("\nView via (Windows 推荐这样做):")
    print(f"  cd {OUT_DIR}")
    print("  python -m http.server 8000")
    print("  http://localhost:8000/calibrated_puff_plume_time.html")


if __name__ == "__main__":
    main()
