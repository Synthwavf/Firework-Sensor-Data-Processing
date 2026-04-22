#!/usr/bin/env python3
"""
Model Validation — Forward Lagrangian Dispersion vs Observed Sensor Data

Quantitative validation of the Gaussian Puff Lagrangian Particle Model
against 5 AeroSpec PM sensors deployed around Lake Union, July 4, 2025.

Approach:
  1. Run the forward dispersion simulation (same physics as the paper model)
  2. At each 1-minute timestep, bilinearly interpolate the modeled concentration
     at each sensor's (x, y) location
  3. Load observed 1-minute-averaged PM data from each sensor
  4. Fit a single global scaling factor (model particle count → µg/m³)
  5. Compute standard air quality model evaluation metrics:
     - R² (coefficient of determination)
     - RMSE (root mean square error, µg/m³)
     - FAC2 (fraction of predictions within factor of 2)
     - FB (fractional bias)
     - NMSE (normalized mean square error)
     - IOA (index of agreement, Willmott 1981)
     - Pearson r (correlation)
  6. Generate validation plots:
     - Time series overlay (modeled vs observed) per sensor
     - Scatter plot (all sensors pooled)
     - Spatial rank comparison at peak time
     - Taylor diagram (optional)

References:
  Chang & Hanna (2004) — Air quality model performance evaluation
  Willmott (1981) — Index of agreement

Output:
  - validation_metrics.csv  — per-sensor and global metrics
  - validation_timeseries.png — time series overlays
  - validation_scatter.png — predicted vs observed scatter
  - validation_spatial_rank.png — peak-time spatial comparison
"""

import numpy as np
import csv
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs', 'validation')

# Source locations (from backward Lagrangian estimation)
SOURCES = {
    'pm25': {'lat': 47.6391, 'lon': -122.3352, 'csv_col': 17, 'label': 'PM$_{2.5}$'},
    'pm1':  {'lat': 47.6389, 'lon': -122.3352, 'csv_col': 16, 'label': 'PM$_1$'},
}

SENSOR_LOCS = {
    1: (47.6378, -122.3295, '2031 Fairview Ave E'),
    2: (47.6469, -122.3263, '2838 Fairview Ave E'),
    3: (47.6493, -122.3316, '2199 N Northlake Way'),
    5: (47.6299, -122.3394, '1200 Westlake Ave N'),
    6: (47.6267, -122.3353, '809 Fairview Pl N'),
}

# Lagrangian model parameters (identical to paper)
SIGMA_H, SIGMA_W = 1.0, 0.3
T_LH, T_LV = 300, 80
DT = 6
RANDOM_SEED = 42

# Emission parameters
EMISSION_START = 22 * 3600 + 20 * 60
EMISSION_END   = 22 * 3600 + 40 * 60
EMISSION_INTERVAL = 30
N_PARTICLES_PER_PUFF = 1500
BURST_HEIGHT_MEAN, BURST_HEIGHT_STD = 120, 35
BURST_RADIAL_SPEED_MEAN = 8.0
BURST_RADIAL_SPEED_STD  = 4.0
BURST_DOWNWARD_BIAS     = 2.0

# Fine grid for validation (need precise interpolation at sensor locations)
GRID_RES    = 15    # meters — finer than heatmap version for accuracy
GRID_EXTENT = 3000  # meters — sensors are within ~1.5 km of source

# Simulation window
SIM_START = 22 * 3600 + 15 * 60
SIM_END   = 23 * 3600 + 20 * 60
SNAPSHOT_INTERVAL = 60   # 1-minute snapshots

# Coordinate conversion
REF_LAT = 47.638
REF_LON = -122.333
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111132.0 * np.cos(np.radians(REF_LAT))

# Counter-to-time
COUNTER_DT = 1.111
T0_SECONDS = 19 * 3600 + 59 * 60 + 30
BG_START, BG_END = 1000, 5000


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def latlon_to_xy(lat, lon):
    return (lon - REF_LON) * M_PER_DEG_LON, (lat - REF_LAT) * M_PER_DEG_LAT

def xy_to_latlon(x, y):
    return REF_LAT + y / M_PER_DEG_LAT, REF_LON + x / M_PER_DEG_LON

def seconds_to_pdt(s):
    h = int(s // 3600); m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"


# ============================================================
# DATA LOADING
# ============================================================
def load_wind_data():
    fp = os.path.join(DATA_DIR, 'KSEA.2025-07-05.csv')
    times, speeds, dirs = [], [], []
    with open(fp, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('Station') or line.startswith(','):
                continue
            parts = line.strip().split(',')
            if len(parts) < 8: continue
            try: ws = float(parts[6]); wd = float(parts[7])
            except: continue
            dt = datetime.strptime(parts[1][:19], '%Y-%m-%dT%H:%M:%S')
            s = dt.hour * 3600 + dt.minute * 60 + dt.second
            if dt.day == 5: s += 86400
            times.append(s); speeds.append(ws); dirs.append(wd)
    return np.array(times), np.array(speeds), np.array(dirs)


def load_sensor_data(csv_col):
    """Load observed PM data, subtract background, return 1-min averages."""
    sensors = {}
    for sid in [1, 2, 3, 5, 6]:
        fp = os.path.join(DATA_DIR, f'Data_{sid}.csv')
        counters, vals = [], []
        with open(fp, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 2: continue
                try: counters.append(int(row[0])); vals.append(float(row[csv_col]))
                except: continue
        ca = np.array(counters); va = np.array(vals)
        if sid == 5 and np.any(np.diff(ca) < -100):
            ca = np.arange(len(ca))
        bg = np.median(va[(ca >= BG_START) & (ca <= BG_END)])
        ts = T0_SECONDS + ca * COUNTER_DT
        net = np.maximum(va - bg, 0)

        # Compute 1-minute averages aligned to snapshot times
        minute_times = np.arange(SIM_START, SIM_END + 1, SNAPSHOT_INTERVAL)
        minute_vals = np.zeros(len(minute_times))
        for j, mt in enumerate(minute_times):
            mask = (ts >= mt - 30) & (ts < mt + 30)
            if np.any(mask):
                minute_vals[j] = np.mean(net[mask])
        sensors[sid] = (minute_times, minute_vals)
    return sensors


# ============================================================
# WIND INTERPOLATION
# ============================================================
def interpolate_wind(t, wt, ws, wd):
    t = np.clip(t, wt[0], wt[-1])
    idx = np.clip(np.searchsorted(wt, t) - 1, 0, len(wt) - 2)
    frac = np.clip((t - wt[idx]) / (wt[idx+1] - wt[idx] + 1e-10), 0, 1)
    speed = ws[idx] + frac * (ws[idx+1] - ws[idx])
    r1, r2 = np.radians(wd[idx]), np.radians(wd[idx+1])
    sd = np.sin(r1) + frac * (np.sin(r2) - np.sin(r1))
    cd = np.cos(r1) + frac * (np.cos(r2) - np.cos(r1))
    dfrom = np.degrees(np.arctan2(sd, cd)) % 360
    dto = (dfrom + 180) % 360
    return speed * np.sin(np.radians(dto)), speed * np.cos(np.radians(dto))


# ============================================================
# FORWARD SIMULATION WITH SENSOR EXTRACTION
# ============================================================
def run_forward_with_sensor_extraction(src_lat, src_lon, is_pm1, wt, ws, wd):
    """
    Run forward Lagrangian simulation. At each 1-minute snapshot,
    extract the modeled concentration at each sensor location
    via bilinear interpolation on the 2D grid.

    Returns:
      model_at_sensors: dict {sensor_id: (time_array, conc_array)}
      where conc is in particle-count-density units (pre-scaling).
    """
    np.random.seed(RANDOM_SEED)
    src_x, src_y = latlon_to_xy(src_lat, src_lon)

    # Grid centered on source
    grid_x = np.arange(src_x - GRID_EXTENT, src_x + GRID_EXTENT, GRID_RES)
    grid_y = np.arange(src_y - GRID_EXTENT, src_y + GRID_EXTENT, GRID_RES)
    nx, ny = len(grid_x), len(grid_y)

    # Sensor positions in grid coordinates
    sensor_xy = {}
    for sid, (slat, slon, _) in SENSOR_LOCS.items():
        sx, sy = latlon_to_xy(slat, slon)
        sensor_xy[sid] = (sx, sy)

    R_h = np.exp(-DT / T_LH)
    R_v = np.exp(-DT / T_LV)

    puff_times = np.arange(EMISSION_START, EMISSION_END, EMISSION_INTERVAL)
    n_puffs = len(puff_times)
    max_p = n_puffs * N_PARTICLES_PER_PUFF

    # Particle arrays
    ax = np.full(max_p, src_x); ay = np.full(max_p, src_y); az = np.zeros(max_p)
    aut = np.zeros(max_p); avt = np.zeros(max_p); awt = np.zeros(max_p)
    active = np.zeros(max_p, dtype=bool)
    rel_t = np.zeros(max_p)

    # Initialize with burst impulse physics
    for i, pt in enumerate(puff_times):
        i0, i1 = i * N_PARTICLES_PER_PUFF, (i+1) * N_PARTICLES_PER_PUFF
        n = N_PARTICLES_PER_PUFF
        az[i0:i1] = np.random.normal(BURST_HEIGHT_MEAN, BURST_HEIGHT_STD, n).clip(30, 250)
        radial_speed = np.abs(np.random.normal(BURST_RADIAL_SPEED_MEAN, BURST_RADIAL_SPEED_STD, n))
        azimuth = np.random.uniform(0, 2 * np.pi, n)
        elevation = np.random.normal(-0.3, 0.5, n)
        horiz_speed = radial_speed * np.cos(elevation)
        vert_speed = radial_speed * np.sin(elevation) - BURST_DOWNWARD_BIAS
        aut[i0:i1] = horiz_speed * np.cos(azimuth)
        avt[i0:i1] = horiz_speed * np.sin(azimuth)
        awt[i0:i1] = vert_speed
        rel_t[i0:i1] = pt

    snapshot_times = list(range(SIM_START, SIM_END + 1, SNAPSHOT_INTERVAL))

    # Storage: model concentration at each sensor for each snapshot
    model_at_sensors = {sid: (np.array(snapshot_times), np.zeros(len(snapshot_times)))
                        for sid in SENSOR_LOCS}
    snap_idx_map = {st: i for i, st in enumerate(snapshot_times)}

    print(f"  Grid: {nx}x{ny}, {n_puffs} puffs, {max_p} particles")
    print(f"  Snapshots: {len(snapshot_times)} at 1-min intervals")

    t = SIM_START
    while t <= SIM_END + 60:
        active |= (rel_t <= t)
        na = np.sum(active)

        if na > 0:
            u_mean, v_mean = interpolate_wind(t, wt, ws, wd)
            nh1 = np.random.randn(na); nh2 = np.random.randn(na); nv = np.random.randn(na)
            aut[active] = R_h * aut[active] + np.sqrt(1 - R_h**2) * SIGMA_H * nh1
            avt[active] = R_h * avt[active] + np.sqrt(1 - R_h**2) * SIGMA_H * nh2
            awt[active] = R_v * awt[active] + np.sqrt(1 - R_v**2) * SIGMA_W * nv
            ax[active] += (u_mean + aut[active]) * DT
            ay[active] += (v_mean + avt[active]) * DT
            az[active] += awt[active] * DT
            if not is_pm1: az[active] -= 0.0001 * DT
            below = active & (az < 0)
            az[below] = np.abs(az[below]); awt[below] = np.abs(awt[below]) * 0.3

        # Capture snapshot
        for st in snapshot_times:
            if abs(t - st) < DT / 2 and st in snap_idx_map:
                si = snap_idx_map.pop(st)

                conc = np.zeros((nx, ny), dtype=np.float64)
                if na > 0:
                    ns = active & (az < 60.0) & (az >= 0)
                    if np.any(ns):
                        xi = ((ax[ns] - grid_x[0]) / GRID_RES).astype(int)
                        yi = ((ay[ns] - grid_y[0]) / GRID_RES).astype(int)
                        v = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                        if np.any(v):
                            np.add.at(conc, (xi[v], yi[v]), 1.0)

                conc_s = gaussian_filter(conc, sigma=2.0)

                # Bilinear interpolation at each sensor location
                for sid, (sx, sy) in sensor_xy.items():
                    # Fractional grid indices
                    fx = (sx - grid_x[0]) / GRID_RES
                    fy = (sy - grid_y[0]) / GRID_RES
                    ix = int(np.floor(fx)); iy = int(np.floor(fy))
                    if 0 <= ix < nx-1 and 0 <= iy < ny-1:
                        dx = fx - ix; dy = fy - iy
                        val = (conc_s[ix, iy] * (1-dx)*(1-dy) +
                               conc_s[ix+1, iy] * dx*(1-dy) +
                               conc_s[ix, iy+1] * (1-dx)*dy +
                               conc_s[ix+1, iy+1] * dx*dy)
                        model_at_sensors[sid][1][si] = val

                time_str = seconds_to_pdt(st)
                if si % 10 == 0:
                    print(f"    {time_str}: {na} active, {np.sum(active & (az < 60)):d} near-surface")

        t += DT

    return model_at_sensors


# ============================================================
# VALIDATION METRICS (Chang & Hanna, 2004)
# ============================================================
def compute_metrics(obs, pred):
    """
    Compute standard air quality model evaluation metrics.
    Only uses timesteps where obs > 0 (avoid dividing by zero in ratio metrics).
    """
    # Use all timesteps for basic metrics
    n = len(obs)
    if n == 0:
        return {}

    o_mean = np.mean(obs)
    p_mean = np.mean(pred)

    # R²
    ss_res = np.sum((obs - pred)**2)
    ss_tot = np.sum((obs - o_mean)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # RMSE
    rmse = np.sqrt(np.mean((obs - pred)**2))

    # Normalized RMSE
    nrmse = rmse / o_mean if o_mean > 0 else float('nan')

    # Pearson correlation
    if np.std(obs) > 0 and np.std(pred) > 0:
        r = np.corrcoef(obs, pred)[0, 1]
    else:
        r = float('nan')

    # --- Metrics requiring obs > 0 ---
    pos = obs > 0.5  # threshold to avoid noise
    o_pos = obs[pos]; p_pos = pred[pos]

    # FAC2: fraction within factor of 2
    if len(o_pos) > 0:
        ratios = p_pos / o_pos
        fac2 = np.mean((ratios >= 0.5) & (ratios <= 2.0))
    else:
        fac2 = float('nan')

    # FB: fractional bias — 2(Ō - P̄)/(Ō + P̄)
    if (o_mean + p_mean) > 0:
        fb = 2 * (o_mean - p_mean) / (o_mean + p_mean)
    else:
        fb = 0.0

    # NMSE: (O - P)² / (Ō · P̄)
    if o_mean * p_mean > 0:
        nmse = np.mean((obs - pred)**2) / (o_mean * p_mean)
    else:
        nmse = float('nan')

    # IOA: Index of Agreement (Willmott, 1981)
    if ss_tot > 0:
        denom = np.sum((np.abs(pred - o_mean) + np.abs(obs - o_mean))**2)
        ioa = 1 - ss_res / denom if denom > 0 else float('nan')
    else:
        ioa = float('nan')

    # MG: geometric mean bias (only for positive pairs)
    if len(o_pos) > 0 and np.all(p_pos > 0):
        mg = np.exp(np.mean(np.log(o_pos)) - np.mean(np.log(p_pos)))
    else:
        mg = float('nan')

    return {
        'N': n,
        'O_mean': o_mean,
        'P_mean': p_mean,
        'R2': r2,
        'r': r,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'FAC2': fac2,
        'FB': fb,
        'NMSE': nmse,
        'IOA': ioa,
        'MG': mg,
    }


def find_optimal_scale(obs_dict, model_dict):
    """
    Find a single global scaling factor α such that
    model_scaled = α * model_raw minimizes total RMSE across all sensors.
    """
    all_obs = []
    all_mod = []
    for sid in obs_dict:
        _, o = obs_dict[sid]
        _, m = model_dict[sid]
        # Only use event + post-event (22:20 onwards, when emissions start)
        mask = obs_dict[sid][0] >= EMISSION_START
        all_obs.extend(o[mask])
        all_mod.extend(m[mask])
    all_obs = np.array(all_obs)
    all_mod = np.array(all_mod)

    # Least-squares optimal: α = Σ(obs*mod) / Σ(mod²)
    denom = np.sum(all_mod**2)
    if denom > 0:
        alpha_ls = np.sum(all_obs * all_mod) / denom
    else:
        alpha_ls = 1.0

    return alpha_ls


# ============================================================
# VISUALIZATION
# ============================================================
def plot_timeseries(obs_dict, model_dict, alpha, pm_type, output_path):
    """Time series overlay: modeled (scaled) vs observed for each sensor."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    sensor_ids = sorted(obs_dict.keys())

    colors = {1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a', 5: '#984ea3', 6: '#ff7f00'}

    for idx, sid in enumerate(sensor_ids):
        ax = axes[idx]
        t_obs, obs = obs_dict[sid]
        t_mod, mod = model_dict[sid]

        # Convert to minutes since 22:00
        t_min_obs = (t_obs - 22*3600) / 60
        t_min_mod = (t_mod - 22*3600) / 60

        ax.plot(t_min_obs, obs, 'o-', color=colors[sid], ms=2, lw=1.5,
                label=f'S{sid} Observed', alpha=0.8)
        ax.plot(t_min_mod, mod * alpha, '--', color='black', lw=1.5,
                label=f'S{sid} Modeled (×{alpha:.2f})', alpha=0.8)

        # Shade emission window
        ax.axvspan(20, 40, alpha=0.1, color='red', label='Emission window' if idx == 0 else None)

        ax.set_ylabel(f'S{sid}\nNet PM (µg/m³)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(15, 80)

        # Compute per-sensor R² for annotation
        mask = t_obs >= EMISSION_START
        o = obs[mask]
        m = (mod * alpha)[mask]
        ss_res = np.sum((o - m)**2)
        ss_tot = np.sum((o - np.mean(o))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rmse = np.sqrt(np.mean((o - m)**2))
        ax.text(0.02, 0.85, f'R²={r2:.3f}  RMSE={rmse:.1f} µg/m³',
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Minutes since 22:00 PDT', fontsize=11)
    fig.suptitle(f'{pm_type} Model Validation — Time Series Comparison\n'
                 f'Forward Lagrangian Dispersion vs AeroSpec Sensor Observations\n'
                 f'Lake Union, Seattle — July 4, 2025',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_scatter(obs_dict, model_dict, alpha, pm_type, output_path):
    """Scatter plot of all predicted vs observed (pooled across sensors)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a', 5: '#984ea3', 6: '#ff7f00'}
    all_obs, all_pred = [], []

    for sid in sorted(obs_dict.keys()):
        t_obs, obs = obs_dict[sid]
        _, mod = model_dict[sid]
        mask = t_obs >= EMISSION_START
        o = obs[mask]
        p = (mod * alpha)[mask]
        ax.scatter(o, p, c=colors[sid], alpha=0.6, s=20, label=f'S{sid}', edgecolors='none')
        all_obs.extend(o); all_pred.extend(p)

    all_obs = np.array(all_obs); all_pred = np.array(all_pred)

    # 1:1 line
    max_val = max(np.max(all_obs), np.max(all_pred)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k-', lw=1.5, label='1:1 line')
    # Factor of 2 lines
    ax.plot([0, max_val], [0, max_val*2], 'k--', lw=0.8, alpha=0.4, label='Factor of 2')
    ax.plot([0, max_val], [0, max_val/2], 'k--', lw=0.8, alpha=0.4)

    # Global metrics
    metrics = compute_metrics(all_obs, all_pred)
    textstr = (f"R² = {metrics['R2']:.3f}\n"
               f"r = {metrics['r']:.3f}\n"
               f"RMSE = {metrics['RMSE']:.2f} µg/m³\n"
               f"FAC2 = {metrics['FAC2']:.3f}\n"
               f"FB = {metrics['FB']:.3f}\n"
               f"IOA = {metrics['IOA']:.3f}\n"
               f"N = {metrics['N']}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Observed Net PM (µg/m³)', fontsize=12)
    ax.set_ylabel('Predicted Net PM (µg/m³)', fontsize=12)
    ax.set_title(f'{pm_type} Model Validation — Scatter Plot\n'
                 f'Forward Lagrangian Dispersion vs Observations',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_spatial_rank(obs_dict, model_dict, alpha, pm_type, output_path):
    """
    Bar chart comparing peak observed vs peak modeled concentration at each sensor.
    Tests whether the model correctly predicts the spatial pattern.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sensor_ids = sorted(obs_dict.keys())
    obs_peaks = []
    mod_peaks = []
    labels = [f'S{sid}' for sid in sensor_ids]

    for sid in sensor_ids:
        _, obs = obs_dict[sid]
        _, mod = model_dict[sid]
        obs_peaks.append(np.max(obs))
        mod_peaks.append(np.max(mod * alpha))

    obs_peaks = np.array(obs_peaks)
    mod_peaks = np.array(mod_peaks)

    # Bar chart of peak values
    x = np.arange(len(sensor_ids))
    w = 0.35
    ax1.bar(x - w/2, obs_peaks, w, label='Observed Peak', color='steelblue', alpha=0.8)
    ax1.bar(x + w/2, mod_peaks, w, label='Modeled Peak', color='coral', alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Peak Net PM (µg/m³)', fontsize=11)
    ax1.set_title('Peak Concentration Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    # Rank comparison
    obs_rank = np.argsort(np.argsort(-obs_peaks)) + 1
    mod_rank = np.argsort(np.argsort(-mod_peaks)) + 1

    ax2.scatter(obs_rank, mod_rank, s=120, c='darkorange', edgecolors='black', zorder=5)
    for i, sid in enumerate(sensor_ids):
        ax2.annotate(f'S{sid}', (obs_rank[i], mod_rank[i]),
                    textcoords='offset points', xytext=(8, 4), fontsize=10)
    ax2.plot([0.5, 5.5], [0.5, 5.5], 'k--', alpha=0.5, label='Perfect rank agreement')
    ax2.set_xlabel('Observed Rank (1=highest)', fontsize=11)
    ax2.set_ylabel('Modeled Rank (1=highest)', fontsize=11)
    ax2.set_title('Spatial Rank Agreement', fontsize=12, fontweight='bold')
    ax2.set_xlim(0.5, 5.5); ax2.set_ylim(0.5, 5.5)
    ax2.set_xticks([1,2,3,4,5]); ax2.set_yticks([1,2,3,4,5])
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(obs_peaks, mod_peaks)
    ax2.text(0.05, 0.85, f'Spearman ρ = {rho:.3f}\np = {p_val:.3f}',
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(f'{pm_type} Spatial Validation — Peak Concentrations\n'
                 f'Lake Union, July 4, 2025',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MODEL VALIDATION")
    print("Forward Lagrangian Dispersion vs Observed Sensor Data")
    print("=" * 60)

    wt, ws, wd = load_wind_data()
    print(f"  Wind data: {len(wt)} records")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for pm_key, pm_cfg in SOURCES.items():
        print(f"\n{'='*60}")
        print(f"  {pm_cfg['label']} VALIDATION")
        print(f"{'='*60}")

        # Load observed data
        print(f"\n  Loading observed {pm_cfg['label']} sensor data...")
        obs_data = load_sensor_data(pm_cfg['csv_col'])
        for sid in sorted(obs_data.keys()):
            _, vals = obs_data[sid]
            print(f"    S{sid}: peak = {np.max(vals):.1f} µg/m³")

        # Run forward model
        print(f"\n  Running forward simulation (source: {pm_cfg['lat']:.4f}°N, "
              f"{abs(pm_cfg['lon']):.4f}°W)...")
        model_data = run_forward_with_sensor_extraction(
            pm_cfg['lat'], pm_cfg['lon'], pm_key == 'pm1', wt, ws, wd
        )

        # Find optimal global scaling factor
        alpha = find_optimal_scale(obs_data, model_data)
        print(f"\n  Optimal scaling factor: α = {alpha:.4f}")
        print(f"  (Maps model particle-density to µg/m³)")

        # Compute per-sensor metrics
        print(f"\n  --- Per-Sensor Metrics (event period, 22:20–23:20) ---")
        print(f"  {'Sensor':>6s}  {'R²':>7s}  {'r':>7s}  {'RMSE':>7s}  {'FAC2':>6s}  "
              f"{'FB':>7s}  {'IOA':>6s}  {'O_peak':>7s}  {'P_peak':>7s}")

        all_obs_pooled = []
        all_pred_pooled = []

        for sid in sorted(obs_data.keys()):
            t_obs, obs = obs_data[sid]
            t_mod, mod = model_data[sid]
            mask = t_obs >= EMISSION_START
            o = obs[mask]
            p = (mod * alpha)[mask]
            m = compute_metrics(o, p)

            all_obs_pooled.extend(o)
            all_pred_pooled.extend(p)

            print(f"  S{sid:>4d}  {m['R2']:>7.3f}  {m['r']:>7.3f}  {m['RMSE']:>7.2f}  "
                  f"{m['FAC2']:>6.3f}  {m['FB']:>7.3f}  {m['IOA']:>6.3f}  "
                  f"{np.max(o):>7.1f}  {np.max(p):>7.1f}")

            all_results.append({
                'PM_type': pm_key,
                'Sensor': f'S{sid}',
                **m,
                'O_peak': np.max(o),
                'P_peak': np.max(p),
                'alpha': alpha,
            })

        # Global pooled metrics
        all_obs_pooled = np.array(all_obs_pooled)
        all_pred_pooled = np.array(all_pred_pooled)
        gm = compute_metrics(all_obs_pooled, all_pred_pooled)
        print(f"\n  --- Global Pooled Metrics ---")
        print(f"  R²={gm['R2']:.3f}  r={gm['r']:.3f}  RMSE={gm['RMSE']:.2f}  "
              f"FAC2={gm['FAC2']:.3f}  FB={gm['FB']:.3f}  IOA={gm['IOA']:.3f}")

        all_results.append({
            'PM_type': pm_key, 'Sensor': 'GLOBAL', **gm,
            'O_peak': np.max(all_obs_pooled),
            'P_peak': np.max(all_pred_pooled),
            'alpha': alpha,
        })

        # Generate plots
        print(f"\n  Generating validation plots...")
        plot_timeseries(obs_data, model_data, alpha, pm_cfg['label'],
                       os.path.join(OUTPUT_DIR, f'validation_timeseries_{pm_key}.png'))
        plot_scatter(obs_data, model_data, alpha, pm_cfg['label'],
                    os.path.join(OUTPUT_DIR, f'validation_scatter_{pm_key}.png'))
        plot_spatial_rank(obs_data, model_data, alpha, pm_cfg['label'],
                         os.path.join(OUTPUT_DIR, f'validation_spatial_rank_{pm_key}.png'))

    # Save metrics to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'validation_metrics.csv')
    fields = ['PM_type', 'Sensor', 'N', 'O_mean', 'P_mean', 'R2', 'r',
              'RMSE', 'NRMSE', 'FAC2', 'FB', 'NMSE', 'IOA', 'MG',
              'O_peak', 'P_peak', 'alpha']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\n  Metrics saved to {csv_path}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
