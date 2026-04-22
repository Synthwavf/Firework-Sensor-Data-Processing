#!/usr/bin/env python3
"""
PM2.5 Forward Dispersion Heatmap — Gaussian Puff Lagrangian Particle Model

Simulates the time-evolving spread of fireworks-generated PM2.5 from the
estimated source location (47.6391°N, 122.3352°W) using the same Lagrangian
stochastic particle physics as the backward source localization model.

Purpose:
  This forward simulation complements the backward source localization by
  visualizing how the fireworks plume spreads from the source through the
  spectator zone over time.  The resulting heatmaps directly illustrate the
  near-field vs far-field concentration gradient — the central finding of
  the journal paper.

Method:
  1. During the fireworks display (22:20–22:40 PDT), particles are released
     every minute from the source at burst height (~100–150 m).
  2. Particles are advected FORWARD through the KSEA wind field with
     Langevin turbulent fluctuations and ground reflection.
  3. At each snapshot time, near-surface particles (z < 50 m) are binned
     onto a 2D grid to produce ground-level concentration maps.
  4. Multi-panel figure shows the plume evolution from 22:15 to 23:15 PDT.

Inputs:
  - Data_1.csv through Data_6.csv (PM2.5 sensor data for validation overlay)
  - KSEA.2025-07-05.csv (5-minute METAR wind data)

Output:
  - Multi-panel time-evolving heatmap figure (PNG)
  - Individual snapshot PNGs
"""

import numpy as np
import csv
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs', 'forward_dispersion_pm25')

# Source location from backward Lagrangian estimation
# (Hanna-D z=200 m configuration from sensitivity sweep; 189 m from known barge)
SOURCE_LAT = 47.6387
SOURCE_LON = -122.3358

# Sensor locations (S4 excluded due to contamination)
SENSOR_LOCS = {
    1: (47.6378, -122.3295),   # 2031 Fairview Ave E
    2: (47.6469, -122.3263),   # 2838 Fairview Ave E
    3: (47.6493, -122.3316),   # 2199 N Northlake Way
    5: (47.6299, -122.3394),   # 1200 Westlake Ave N
    6: (47.6267, -122.3353),   # 809 Fairview Pl N
}

# Lagrangian model parameters (2D, same as backward model)
# Derived from KSEA METAR via Turner1964 -> log-law -> van Ulden-Holtslag 1985
# -> Zilitinkevich 1972 -> Hanna 1982 NEUTRAL class at z=200 m.
# 2D formulation: no vertical coordinate (no reliable vertical wind from KSEA,
# source and receptor both within a thin near-surface layer).  The old z<50 m
# "breathing zone" filter is replaced by an exponential residence-time decay
# TAU_RES that accounts for smoke escaping the near-surface layer.
SIGMA_H    = 0.33    # horizontal turbulent velocity std (m/s)  [Hanna-D, z=200 m]
T_LH       = 148     # horizontal Lagrangian integral timescale (s)  [Hanna-D]
TAU_RES    = 500     # vertical-escape residence time (s)  [Hanna-D equivalent]
DT         = 6       # integration time step (s)  -- T_LH/DT = 24.7, well resolved
RANDOM_SEED = 42

# Emission parameters (horizontal only; no burst-height distribution in 2D)
EMISSION_START  = 22 * 3600 + 20 * 60   # 22:20 PDT in seconds
EMISSION_END    = 22 * 3600 + 40 * 60   # 22:40 PDT in seconds
EMISSION_INTERVAL = 60                    # release a puff every 60 seconds
N_PARTICLES_PER_PUFF = 2000              # particles per puff release

# Grid parameters for heatmap
GRID_RES    = 20      # meters per cell
GRID_EXTENT = 2500    # meters from source in each direction

# Snapshot times (PDT, in seconds from midnight)
SNAPSHOT_TIMES_PDT = [
    22 * 3600 + 15 * 60,   # 22:15 — pre-event baseline
    22 * 3600 + 22 * 60,   # 22:22 — early fireworks
    22 * 3600 + 28 * 60,   # 22:28 — developing plume
    22 * 3600 + 35 * 60,   # 22:35 — mid-display
    22 * 3600 + 42 * 60,   # 22:42 — display ending
    22 * 3600 + 50 * 60,   # 22:50 — post-display spread
    23 * 3600 + 0 * 60,    # 23:00 — continued transport
    23 * 3600 + 15 * 60,   # 23:15 — late transport
]

# Coordinate conversion
REF_LAT = 47.638
REF_LON = -122.333
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111132.0 * np.cos(np.radians(REF_LAT))

# Counter-to-time conversion (for sensor data)
COUNTER_DT = 1.111
T0_SECONDS = 19 * 3600 + 59 * 60 + 30  # counter 0 ≈ 19:59:30 PDT
BG_START   = 1000
BG_END     = 5000


# ============================================================
# COORDINATE FUNCTIONS
# ============================================================
def latlon_to_xy(lat, lon):
    """Convert lat/lon to local meters (x=east, y=north)."""
    x = (lon - REF_LON) * M_PER_DEG_LON
    y = (lat - REF_LAT) * M_PER_DEG_LAT
    return x, y

def xy_to_latlon(x, y):
    """Convert local meters back to lat/lon."""
    lat = REF_LAT + y / M_PER_DEG_LAT
    lon = REF_LON + x / M_PER_DEG_LON
    return lat, lon

def seconds_to_pdt_str(s):
    """Convert seconds-from-midnight to 'HH:MM' PDT string."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"


# Precompute positions
SOURCE_XY = latlon_to_xy(SOURCE_LAT, SOURCE_LON)
SENSOR_XY = {sid: latlon_to_xy(*ll) for sid, ll in SENSOR_LOCS.items()}


# ============================================================
# DATA LOADING
# ============================================================
def load_wind_data(data_dir):
    """Load KSEA 5-minute METAR wind data."""
    filepath = os.path.join(data_dir, 'KSEA.2025-07-05.csv')
    times, speeds, directions = [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('Station') or line.startswith(','):
                continue
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            try:
                wind_speed = float(parts[6])
                wind_direction = float(parts[7])
            except (ValueError, IndexError):
                continue

            dt = datetime.strptime(parts[1][:19], '%Y-%m-%dT%H:%M:%S')
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            if dt.day == 5:
                seconds += 86400

            times.append(seconds)
            speeds.append(wind_speed)
            directions.append(wind_direction)

    return np.array(times), np.array(speeds), np.array(directions)


def load_sensor_pm25(data_dir):
    """
    Load PM2.5 sensor data for validation overlay.
    Returns dict: {sensor_id: (time_seconds_array, net_pm25_array)}
    where time is in PDT seconds from midnight.
    """
    sensors = {}
    for sid in [1, 2, 3, 5, 6]:
        filepath = os.path.join(data_dir, f'Data_{sid}.csv')
        counters, pm25_values = [], []

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 2:
                    continue
                try:
                    counters.append(int(row[0]))
                    pm25_values.append(float(row[17]))
                except (ValueError, IndexError):
                    continue

        counter_arr = np.array(counters)
        pm25_arr = np.array(pm25_values)

        if sid == 5:
            diffs = np.diff(counter_arr)
            if np.any(diffs < -100):
                counter_arr = np.arange(len(counter_arr))

        # Compute background
        bg_mask = (counter_arr >= BG_START) & (counter_arr <= BG_END)
        background = np.median(pm25_arr[bg_mask])

        # Convert counter to time
        time_seconds = T0_SECONDS + counter_arr * COUNTER_DT
        net_pm25 = np.maximum(pm25_arr - background, 0)

        sensors[sid] = (time_seconds, net_pm25)

    return sensors


# ============================================================
# WIND INTERPOLATION
# ============================================================
def interpolate_wind(t, wind_times, wind_speeds, wind_dirs):
    """
    Interpolate wind at time t. Returns (u_east, v_north) transport velocity.
    """
    t = np.clip(t, wind_times[0], wind_times[-1])
    idx = np.clip(np.searchsorted(wind_times, t) - 1, 0, len(wind_times) - 2)
    frac = np.clip(
        (t - wind_times[idx]) / (wind_times[idx + 1] - wind_times[idx] + 1e-10),
        0, 1
    )

    speed = wind_speeds[idx] + frac * (wind_speeds[idx + 1] - wind_speeds[idx])

    r1 = np.radians(wind_dirs[idx])
    r2 = np.radians(wind_dirs[idx + 1])
    sin_d = np.sin(r1) + frac * (np.sin(r2) - np.sin(r1))
    cos_d = np.cos(r1) + frac * (np.cos(r2) - np.cos(r1))
    direction_from = np.degrees(np.arctan2(sin_d, cos_d)) % 360
    direction_toward = (direction_from + 180) % 360

    u_east  = speed * np.sin(np.radians(direction_toward))
    v_north = speed * np.cos(np.radians(direction_toward))
    return u_east, v_north


# ============================================================
# FORWARD LAGRANGIAN DISPERSION
# ============================================================
def run_forward_dispersion(wind_times, wind_speeds, wind_dirs):
    """
    Forward Lagrangian particle dispersion from the estimated source.

    Returns:
      snapshots: dict {time_seconds: concentration_2d_array}
      grid_x, grid_y: 1D arrays of grid cell centers (meters)
    """
    np.random.seed(RANDOM_SEED)

    # Set up grid centered on SOURCE
    src_x, src_y = SOURCE_XY
    grid_x = np.arange(src_x - GRID_EXTENT, src_x + GRID_EXTENT, GRID_RES)
    grid_y = np.arange(src_y - GRID_EXTENT, src_y + GRID_EXTENT, GRID_RES)
    nx, ny = len(grid_x), len(grid_y)

    # Langevin autocorrelation coefficient (horizontal only)
    R_h = np.exp(-DT / T_LH)

    # Simulation time range
    sim_start = SNAPSHOT_TIMES_PDT[0]          # 22:15
    sim_end   = SNAPSHOT_TIMES_PDT[-1] + 60    # 23:16 (a bit past last snapshot)

    # Pre-compute puff release times
    puff_times = np.arange(EMISSION_START, EMISSION_END, EMISSION_INTERVAL)
    n_puffs = len(puff_times)

    # Storage for all particles (2D: x, y, u_turb, v_turb, release_time)
    max_particles = n_puffs * N_PARTICLES_PER_PUFF
    all_x = np.zeros(max_particles)
    all_y = np.zeros(max_particles)
    all_ut = np.zeros(max_particles)
    all_vt = np.zeros(max_particles)
    all_active = np.zeros(max_particles, dtype=bool)
    all_release_time = np.zeros(max_particles)

    # Pre-fill release positions
    for i, pt in enumerate(puff_times):
        i0 = i * N_PARTICLES_PER_PUFF
        i1 = i0 + N_PARTICLES_PER_PUFF
        all_x[i0:i1] = src_x
        all_y[i0:i1] = src_y
        all_ut[i0:i1] = np.random.normal(0, SIGMA_H, N_PARTICLES_PER_PUFF)
        all_vt[i0:i1] = np.random.normal(0, SIGMA_H, N_PARTICLES_PER_PUFF)
        all_release_time[i0:i1] = pt

    # Sort snapshot times for efficient capture
    snap_set = set(SNAPSHOT_TIMES_PDT)
    snapshots = {}

    print(f"  Forward simulation: {sim_start} to {sim_end} s PDT")
    print(f"  Puffs: {n_puffs} releases, {N_PARTICLES_PER_PUFF} particles each")
    print(f"  Total particles: {max_particles}")
    print(f"  Grid: {nx} x {ny} cells at {GRID_RES}m resolution")

    # --- Time-stepping loop ---
    t = sim_start
    step_count = 0

    while t <= sim_end:
        # Activate newly released particles
        newly_active = (~all_active) & (all_release_time <= t) & (all_release_time > t - DT - 0.1)
        all_active |= (all_release_time <= t)

        active = all_active
        n_active = np.sum(active)

        if n_active > 0:
            # Get mean wind
            u_mean, v_mean = interpolate_wind(t, wind_times, wind_speeds, wind_dirs)

            # Update horizontal turbulent velocities (Langevin AR1)
            noise_h1 = np.random.randn(n_active)
            noise_h2 = np.random.randn(n_active)

            all_ut[active] = R_h * all_ut[active] + np.sqrt(1 - R_h**2) * SIGMA_H * noise_h1
            all_vt[active] = R_h * all_vt[active] + np.sqrt(1 - R_h**2) * SIGMA_H * noise_h2

            # Forward step: ADD velocity (horizontal only)
            all_x[active] += (u_mean + all_ut[active]) * DT
            all_y[active] += (v_mean + all_vt[active]) * DT

        # --- Capture snapshot ---
        # Find the closest snapshot time within DT/2
        for snap_t in SNAPSHOT_TIMES_PDT:
            if abs(t - snap_t) < DT / 2 and snap_t not in snapshots:
                conc = np.zeros((nx, ny), dtype=np.float64)
                if n_active > 0:
                    # Residence-time weighting: particles released long ago
                    # have likely escaped the near-surface layer.
                    ages = snap_t - all_release_time[active]
                    weights = np.exp(-np.maximum(ages, 0.0) / TAU_RES)
                    xi = ((all_x[active] - grid_x[0]) / GRID_RES).astype(int)
                    yi = ((all_y[active] - grid_y[0]) / GRID_RES).astype(int)
                    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                    if np.any(valid):
                        np.add.at(conc, (xi[valid], yi[valid]), weights[valid])

                # Smooth for visualization
                conc_smooth = gaussian_filter(conc, sigma=3.0)
                snapshots[snap_t] = conc_smooth
                print(f"    Snapshot at {seconds_to_pdt_str(snap_t)} PDT: "
                      f"{n_active} active particles")

        t += DT
        step_count += 1

    print(f"  Completed {step_count} time steps")
    return snapshots, grid_x, grid_y


# ============================================================
# VISUALIZATION
# ============================================================
def make_multipanel_figure(snapshots, grid_x, grid_y, sensor_data,
                           wind_times, wind_speeds, wind_dirs, output_path):
    """Generate multi-panel heatmap figure showing plume evolution."""

    n_panels = len(SNAPSHOT_TIMES_PDT)
    n_cols = 4
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 13))
    axes = axes.flatten()

    # Global normalization for consistent color scale
    all_max = max(np.max(snapshots[t]) for t in SNAPSHOT_TIMES_PDT if t in snapshots)
    if all_max == 0:
        all_max = 1.0

    src_x, src_y = SOURCE_XY

    for idx, snap_t in enumerate(SNAPSHOT_TIMES_PDT):
        ax = axes[idx]

        if snap_t in snapshots:
            conc = snapshots[snap_t]
            conc_norm = conc / all_max
        else:
            conc_norm = np.zeros((len(grid_x), len(grid_y)))

        im = ax.imshow(conc_norm.T, origin='lower',
                       extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                       cmap='YlOrRd', aspect='equal', vmin=0, vmax=1,
                       interpolation='bilinear')

        # Source marker
        ax.plot(src_x, src_y, '*', color='lime', ms=15, mec='black', mew=1.5,
                zorder=20)

        # Sensor markers with observed concentration at this time
        for sid, (sx, sy) in SENSOR_XY.items():
            ax.plot(sx, sy, '^', color='dodgerblue', ms=8, mec='white', mew=1, zorder=15)

            # Get observed net PM2.5 at this snapshot time (1-min average)
            if sid in sensor_data:
                t_arr, pm_arr = sensor_data[sid]
                time_mask = (t_arr >= snap_t - 30) & (t_arr <= snap_t + 30)
                if np.any(time_mask):
                    obs_val = np.mean(pm_arr[time_mask])
                else:
                    obs_val = 0.0
                ax.annotate(f'S{sid}\n{obs_val:.0f}',
                           (sx, sy), textcoords='offset points',
                           xytext=(8, 6), fontsize=7, color='white',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15',
                                    facecolor='black', alpha=0.7))

        # Wind arrow
        u_w, v_w = interpolate_wind(snap_t, wind_times, wind_speeds, wind_dirs)
        arrow_scale = 200
        ax.annotate('', xy=(src_x + u_w * arrow_scale, src_y + v_w * arrow_scale),
                    xytext=(src_x, src_y),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                    zorder=18)

        # Title
        time_str = seconds_to_pdt_str(snap_t)
        if snap_t < EMISSION_START:
            phase = "Pre-Event"
        elif snap_t <= EMISSION_END:
            phase = "Active Display"
        elif snap_t <= EMISSION_END + 10 * 60:
            phase = "Post-Display"
        else:
            phase = "Late Transport"
        ax.set_title(f'{time_str} PDT — {phase}', fontsize=11, fontweight='bold')

        # Axis labels
        ax.set_xlabel('East-West (m)', fontsize=8)
        ax.set_ylabel('North-South (m)', fontsize=8)
        ax.tick_params(labelsize=7)

        # Zoom to relevant area
        ax.set_xlim(src_x - 2000, src_x + 2000)
        ax.set_ylim(src_y - 2000, src_y + 2000)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Relative PM$_{2.5}$ Concentration\n(Forward Dispersion Model)',
                   fontsize=11)

    fig.suptitle(
        'PM$_{2.5}$ Forward Dispersion Heatmap — Fireworks Plume Evolution\n'
        f'Source: {SOURCE_LAT:.4f}°N, {abs(SOURCE_LON):.4f}°W  |  '
        f'Lake Union, Seattle  |  July 4, 2025\n'
        f'Gaussian Puff Lagrangian Model  |  KSEA Wind  |  '
        f'★ = Source  |  ▲ = AeroSpec Sensor (observed net µg/m³)',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Multi-panel figure saved to {output_path}")


def make_individual_snapshots(snapshots, grid_x, grid_y, sensor_data,
                               wind_times, wind_speeds, wind_dirs, output_dir):
    """Generate individual high-resolution snapshot figures."""

    all_max = max(np.max(snapshots[t]) for t in SNAPSHOT_TIMES_PDT if t in snapshots)
    if all_max == 0:
        all_max = 1.0

    src_x, src_y = SOURCE_XY

    for snap_t in SNAPSHOT_TIMES_PDT:
        if snap_t not in snapshots:
            continue

        conc = snapshots[snap_t]
        conc_norm = conc / all_max

        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(conc_norm.T, origin='lower',
                       extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                       cmap='YlOrRd', aspect='equal', vmin=0, vmax=1,
                       interpolation='bilinear')

        # Source
        ax.plot(src_x, src_y, '*', color='lime', ms=22, mec='black', mew=2, zorder=20,
                label=f'Source ({SOURCE_LAT:.4f}°N, {abs(SOURCE_LON):.4f}°W)')

        # Sensors
        for sid, (sx, sy) in SENSOR_XY.items():
            ax.plot(sx, sy, '^', color='dodgerblue', ms=12, mec='white', mew=1.5, zorder=15)
            if sid in sensor_data:
                t_arr, pm_arr = sensor_data[sid]
                time_mask = (t_arr >= snap_t - 30) & (t_arr <= snap_t + 30)
                obs_val = np.mean(pm_arr[time_mask]) if np.any(time_mask) else 0.0
                ax.annotate(f'S{sid}: {obs_val:.1f} µg/m³',
                           (sx, sy), textcoords='offset points',
                           xytext=(12, 10), fontsize=10, color='white',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='black', alpha=0.7))

        # Wind arrow
        u_w, v_w = interpolate_wind(snap_t, wind_times, wind_speeds, wind_dirs)
        arrow_scale = 250
        ax.annotate('', xy=(src_x + u_w * arrow_scale, src_y + v_w * arrow_scale),
                    xytext=(src_x, src_y),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=3),
                    zorder=18)

        # Distance rings (250m, 500m, 1000m)
        for r in [250, 500, 1000, 1500]:
            circle = plt.Circle((src_x, src_y), r, fill=False,
                              color='white', alpha=0.3, linestyle='--', lw=1)
            ax.add_patch(circle)
            ax.text(src_x + r * 0.707, src_y + r * 0.707, f'{r}m',
                   fontsize=7, color='white', alpha=0.5)

        time_str = seconds_to_pdt_str(snap_t)
        ax.set_title(f'PM$_{{2.5}}$ Forward Dispersion — {time_str} PDT\n'
                     f'Source: {SOURCE_LAT:.4f}°N, {abs(SOURCE_LON):.4f}°W',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('East-West (m from reference)', fontsize=11)
        ax.set_ylabel('North-South (m from reference)', fontsize=11)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

        plt.colorbar(im, ax=ax,
                     label='Relative PM$_{2.5}$ concentration',
                     shrink=0.8)

        ax.set_xlim(src_x - 2000, src_x + 2000)
        ax.set_ylim(src_y - 2000, src_y + 2000)
        ax.grid(True, alpha=0.15, color='white')

        plt.tight_layout()
        fname = f'pm25_dispersion_{time_str.replace(":", "")}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Individual snapshots saved to {output_dir}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PM2.5 Forward Dispersion Heatmap")
    print("Gaussian Puff Lagrangian Particle Model")
    print("=" * 60)

    # Load wind data
    print("\nLoading KSEA wind data...")
    wind_times, wind_speeds, wind_dirs = load_wind_data(DATA_DIR)
    event_mask = (wind_times >= 21 * 3600) & (wind_times <= 23.5 * 3600)
    print(f"  {len(wind_times)} records, event mean: FROM "
          f"{np.mean(wind_dirs[event_mask]):.1f}° at "
          f"{np.mean(wind_speeds[event_mask]):.2f} m/s")

    # Load sensor data for validation overlay
    print("\nLoading PM2.5 sensor data for overlay...")
    sensor_data = load_sensor_pm25(DATA_DIR)
    for sid in sorted(sensor_data.keys()):
        t_arr, pm_arr = sensor_data[sid]
        event_vals = pm_arr[(t_arr >= EMISSION_START) & (t_arr <= EMISSION_END + 3600)]
        print(f"  S{sid}: peak net = {np.max(event_vals):.1f} µg/m³")

    # Source location
    src_x, src_y = SOURCE_XY
    print(f"\nSource location: {SOURCE_LAT:.4f}°N, {abs(SOURCE_LON):.4f}°W")
    print(f"  Local coords: ({src_x:.0f}, {src_y:.0f}) m")

    # Run forward dispersion
    print("\nRunning forward Lagrangian dispersion...")
    snapshots, grid_x, grid_y = run_forward_dispersion(
        wind_times, wind_speeds, wind_dirs
    )

    # Generate figures
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nGenerating multi-panel figure...")
    make_multipanel_figure(
        snapshots, grid_x, grid_y, sensor_data,
        wind_times, wind_speeds, wind_dirs,
        os.path.join(OUTPUT_DIR, 'pm25_forward_dispersion_heatmap.png')
    )

    print("\nGenerating individual snapshots...")
    make_individual_snapshots(
        snapshots, grid_x, grid_y, sensor_data,
        wind_times, wind_speeds, wind_dirs,
        OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("DONE — PM2.5 Forward Dispersion Heatmap")
    print("=" * 60)
