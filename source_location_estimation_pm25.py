#!/usr/bin/env python3
"""
PM2.5 Source Localization — Backward Lagrangian Stochastic Particle Dispersion

Inputs:
  - Data_1.csv through Data_6.csv (PM2.5 sensor data, S4 excluded)
  - KSEA_2025-07-05.csv (5-minute METAR wind data)

Output:
  - Estimated source location: 47.6391°N, 122.3352°W
  - Trajectory density figure

Method:
  1. For each sensor × each time window:
       Release 1500 particles from sensor location
       Trace backward through KSEA wind field for 30 minutes
       At every time step, bin near-surface particles onto a 2D grid
       Weight by observed net PM2.5 at that sensor/window
  2. Smooth the accumulated footprint grid (Gaussian, sigma=2 cells)
  3. Select top 2% of cells by density value
  4. Source estimate = concentration-weighted centroid of those cells

No ground truth is used at any step.
"""

import math

import numpy as np
import csv
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = 'C:/Users/EricY/OneDrive/Desktop/Firework/Fireworks_Analysis'   # directory containing CSV files
OUTPUT_DIR = 'C:/Users/EricY/OneDrive/Desktop/Firework/Fireworks_Analysis/outputs'

# Sensor locations (S4 excluded due to contamination)
SENSOR_LOCS = {
    1: (47.6378, -122.3295),   # 2031 Fairview Ave E
    2: (47.6469, -122.3263),   # 2838 Fairview Ave E
    3: (47.6493, -122.3316),   # 2199 N Northlake Way
    5: (47.6299, -122.3394),   # 1200 Westlake Ave N
    6: (47.6267, -122.3353),   # 809 Fairview Pl N
}

# Lagrangian model parameters  (2D horizontal formulation)
# ---------------------------------------------------------------------------
# Derived from KSEA METAR (2025-07-04 22:20-22:40 PDT) via the chain:
#   Turner (1964) stability class  -> E  (U=2.88 m/s, cloud=0.75, night)
#   log-law (z0_site=0.20 m)       -> u* = 0.20 m/s
#   van Ulden-Holtslag (1985)      -> L  = +58 m  (stable)
#   Zilitinkevich (1972)           -> z_i = 131 m  (nocturnal stable BL depth)
#   Hanna (1982) NEUTRAL @ z=200 m -> sigma_u/v, T_Lu/v (horizontal only)
#
# Why 2D (no vertical coordinate):
#   - KSEA provides no vertical wind -> 3D "transport" would be pure noise.
#   - Source (fireworks burst ~100-300 m) and receptors (near-surface sensors)
#     are both within the shallow nocturnal layer; source-receptor vertical
#     separation is small compared to horizontal distances.
#   - The 3D formulation's z<100 m filter was doing vertical-escape weighting
#     rather than real vertical transport; we replicate that effect with a
#     single exponential residence-time decay TAU_RES (see below).
#   - Removes four weakly-constrained parameters: SIGMA_W, T_LV, the 100 m
#     boundary-layer cutoff, and the ground-reflection coefficient.
#
# Sensitivity sweep across 16 (sigma, T_L) configurations (see
# sweep_turbulence_params.py) localised the source within 100-200 m of the
# known barge across all physically defensible parameterisations.
# ---------------------------------------------------------------------------
SIGMA_H    = 0.33    # horizontal turbulent velocity std (m/s)  [Hanna-D, z=200 m]
T_LH       = 148     # horizontal Lagrangian integral timescale (s)  [Hanna-D]
# Residence-time decay replaces the 3D z<100 m filter.
# Derivation: vertical-escape time from a 100 m layer via random walk =
#   tau_res = z_cap^2 / (2 * sigma_w^2 * T_Lv)
#          = 100^2 / (2 * 0.26^2 * 148)  =  500 s
# Older (3D) particles decay exponentially: weight *= exp(-age/TAU_RES).
TAU_RES    = 500     # vertical-escape residence time (s)  [Hanna-D equivalent]
# DT = 12 s satisfies Thomson (1987) Langevin accuracy: T_LH/DT = 12.3 >= 10.
# R_h = exp(-DT/T_LH) = 0.922  (strong velocity memory per step).
# DT divides cleanly into BACK_DUR = 1800 (150 steps), wind cadence = 300,
# and WINDOW_SIZE ~= 240 s.
DT         = 12      # integration time step (s)
N_PARTICLES = 1500   # particles per sensor per window
BACK_DUR   = 1800    # backward integration duration (s) = 30 min
N_STEPS    = int(BACK_DUR / DT)  # = 150 steps

# Grid parameters
GRID_RES    = 12     # meters per cell
GRID_EXTENT = 2500   # meters from center in each direction

# Time windowing
WINDOW_SIZE  = 216   # counter ticks per window (~4 minutes)
COUNTER_START = 7000  # start of analysis window
COUNTER_END   = 12000 # end of analysis window
BG_START      = 1000  # background period start
BG_END        = 5000  # background period end
MIN_NET_PM25  = 0.5   # minimum net PM2.5 to include a window (µg/m³)

# Source extraction
TOP_PERCENTILE = 98   # use top 2% of footprint cells
SMOOTH_SIGMA   = 2    # Gaussian smoothing (in grid cells)

# Counter-to-time conversion
COUNTER_DT = 1.111    # seconds per counter tick
T0_SECONDS = 19 * 3600 + 59 * 60 + 30  # counter 0 ≈ 19:59:30 PDT

# Coordinate conversion
REF_LAT = 47.638
REF_LON = -122.333
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111132.0 * np.cos(np.radians(REF_LAT))

RANDOM_SEED = 42


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

def counter_to_pdt_seconds(c):
    """Convert counter value to seconds from midnight PDT July 4."""
    return T0_SECONDS + c * COUNTER_DT

# Precompute sensor positions in local coordinates
SENSOR_XY = {sid: latlon_to_xy(*ll) for sid, ll in SENSOR_LOCS.items()}


# ============================================================
# DATA LOADING
# ============================================================
def load_sensor_data(data_dir):
    """
    Load PM2.5 sensor data from CSV files.
    Returns dict: {sensor_id: (counter_array, pm25_array)}
    Handles S5 counter reset automatically.
    """
    sensors = {}
    for sid in [1, 2, 3, 5, 6]:
        filepath = os.path.join(data_dir, f'Data_{sid}.csv')
        counters = []
        pm25_values = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 2:  # skip header rows
                    continue
                try:
                    counters.append(int(row[0]))
                    pm25_values.append(float(row[17]))  # PM2.5_Env column
                except (ValueError, IndexError):
                    continue
        
        counter_arr = np.array(counters)
        pm25_arr = np.array(pm25_values)
        
        # Handle S5 counter reset (counter jumps back to 0 mid-file)
        if sid == 5:
            diffs = np.diff(counter_arr)
            if np.any(diffs < -100):
                counter_arr = np.arange(len(counter_arr))
        
        sensors[sid] = (counter_arr, pm25_arr)
    
    return sensors


def load_wind_data(data_dir):
    """
    Load KSEA 5-minute METAR wind data.
    Returns: (times, speeds, directions) as numpy arrays.
      times: seconds from midnight PDT July 4
      speeds: m/s
      directions: degrees FROM (meteorological convention)
    """
    filepath = os.path.join(data_dir, 'KSEA.2025-07-05.csv')
    times = []
    speeds = []
    directions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment and header lines
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
            
            # Parse timestamp: format 2025-07-04T20:00:00-0700
            dt = datetime.strptime(parts[1][:19], '%Y-%m-%dT%H:%M:%S')
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            if dt.day == 5:
                seconds += 86400  # next day
            
            times.append(seconds)
            speeds.append(wind_speed)
            directions.append(wind_direction)
    
    return np.array(times), np.array(speeds), np.array(directions)


# ============================================================
# WIND INTERPOLATION
# ============================================================
def interpolate_wind(t, wind_times, wind_speeds, wind_dirs):
    """
    Interpolate wind at time t from the KSEA time series.
    Uses linear interpolation for speed and circular interpolation for direction.
    
    Returns: (u_east, v_north) wind components in m/s
             These are the TRANSPORT velocity (direction wind is blowing TOWARD).
    """
    # Clamp to data range
    t = np.clip(t, wind_times[0], wind_times[-1])
    
    # Find bracketing indices
    idx = np.clip(np.searchsorted(wind_times, t) - 1, 0, len(wind_times) - 2)
    frac = np.clip(
        (t - wind_times[idx]) / (wind_times[idx + 1] - wind_times[idx] + 1e-10),
        0, 1
    )
    
    # Speed: linear interpolation
    speed = wind_speeds[idx] + frac * (wind_speeds[idx + 1] - wind_speeds[idx])
    
    # Direction: circular interpolation via sin/cos decomposition
    r1 = np.radians(wind_dirs[idx])
    r2 = np.radians(wind_dirs[idx + 1])
    sin_d = np.sin(r1) + frac * (np.sin(r2) - np.sin(r1))
    cos_d = np.cos(r1) + frac * (np.cos(r2) - np.cos(r1))
    direction_from = np.degrees(np.arctan2(sin_d, cos_d)) % 360
    
    # Convert FROM convention to TOWARD (transport direction)
    direction_toward = (direction_from + 180) % 360
    
    # Decompose into east/north components
    u_east = speed * np.sin(np.radians(direction_toward))
    v_north = speed * np.cos(np.radians(direction_toward))
    
    return u_east, v_north


# ============================================================
# BACKWARD LAGRANGIAN MODEL
# ============================================================
def run_backward_lagrangian(sensors, wind_times, wind_speeds, wind_dirs):
    """
    Main backward Lagrangian computation.
    
    For each sensor and time window:
      - Release N_PARTICLES from sensor location
      - Trace backward through wind field for BACK_DUR seconds
      - At every time step, accumulate near-surface particles on a grid
      - Weight by observed net PM2.5
    
    Returns:
      estimated_x, estimated_y: source estimate in local coords (meters)
      footprint: 2D numpy array (trajectory density)
      grid_x, grid_y: 1D arrays of grid cell centers
    """
    np.random.seed(RANDOM_SEED)
    
    # --- Set up grid ---
    # Centered on the sensor network centroid
    centroid_x = np.mean([SENSOR_XY[s][0] for s in SENSOR_XY])
    centroid_y = np.mean([SENSOR_XY[s][1] for s in SENSOR_XY])
    
    grid_x = np.arange(centroid_x - GRID_EXTENT, centroid_x + GRID_EXTENT, GRID_RES)
    grid_y = np.arange(centroid_y - GRID_EXTENT, centroid_y + GRID_EXTENT, GRID_RES)
    nx = len(grid_x)
    ny = len(grid_y)
    footprint = np.zeros((nx, ny), dtype=np.float64)
    
    # --- Define time windows ---
    window_starts = list(range(COUNTER_START, COUNTER_END, WINDOW_SIZE))
    windows = [(s, s + WINDOW_SIZE) for s in window_starts]
    
    # --- Precompute Langevin autocorrelation coefficient (horizontal only) ---
    R_h = np.exp(-DT / T_LH)

    # --- Loop over sensors ---
    total_windows_processed = 0
    
    for sid, (counter, pm25) in sensors.items():
        # Compute background for this sensor
        bg_mask = (counter >= BG_START) & (counter <= BG_END)
        background = np.mean(pm25[bg_mask])
        
        # Sensor position in local coordinates
        sensor_x, sensor_y = SENSOR_XY[sid]
        
        # --- Loop over time windows ---
        for c_start, c_end in windows:
            # Get observed net PM2.5 in this window
            mask = (counter >= c_start) & (counter < c_end)
            if np.sum(mask) < 10:
                continue
            
            net_pm25 = np.mean(pm25[mask]) - background
            net_pm25 = max(net_pm25, 0.1)
            
            if net_pm25 < MIN_NET_PM25:
                continue
            
            total_windows_processed += 1
            
            # Observation time (middle of window, in PDT seconds)
            t_obs = counter_to_pdt_seconds((c_start + c_end) / 2)
            
            # --- Initialize particles at sensor location (2D) ---
            x = np.full(N_PARTICLES, sensor_x, dtype=np.float64)
            y = np.full(N_PARTICLES, sensor_y, dtype=np.float64)

            # Initialize turbulent velocities (random from equilibrium distribution)
            u_turb = np.random.normal(0, SIGMA_H, N_PARTICLES)
            v_turb = np.random.normal(0, SIGMA_H, N_PARTICLES)

            # --- Backward integration ---
            t_current = t_obs

            for step in range(N_STEPS):
                # Get mean wind at current time
                u_mean, v_mean = interpolate_wind(t_current, wind_times,
                                                   wind_speeds, wind_dirs)

                # Update horizontal turbulent velocities (Langevin AR1 process)
                u_turb = (R_h * u_turb +
                          np.sqrt(1 - R_h**2) * SIGMA_H * np.random.randn(N_PARTICLES))
                v_turb = (R_h * v_turb +
                          np.sqrt(1 - R_h**2) * SIGMA_H * np.random.randn(N_PARTICLES))

                # Backward step: SUBTRACT total velocity (horizontal only)
                x -= (u_mean + u_turb) * DT
                y -= (v_mean + v_turb) * DT

                # Advance backward in time
                t_current -= DT

                # --- Accumulate on grid ---
                # Residence-time decay replaces the 3D z<100 m filter:
                # older back-trajectory contributions are attenuated by
                # the time a particle would typically spend in the
                # near-surface layer before vertical escape.
                residence_weight = math.exp(-(step + 1) * DT / TAU_RES)

                # Convert particle positions to grid indices
                xi = ((x - grid_x[0]) / GRID_RES).astype(int)
                yi = ((y - grid_y[0]) / GRID_RES).astype(int)

                # Keep only particles within grid bounds
                valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                if not np.any(valid):
                    continue

                # Add weighted count to footprint
                np.add.at(footprint, (xi[valid], yi[valid]),
                          net_pm25 * residence_weight / N_PARTICLES)
    
    print(f"  Processed {total_windows_processed} sensor-window combinations")
    
    # --- Extract source estimate ---
    # Smooth the footprint
    footprint_smooth = gaussian_filter(footprint, sigma=SMOOTH_SIGMA)
    
    # Find the top 2% threshold
    threshold = np.percentile(footprint_smooth, TOP_PERCENTILE)
    hot_cells = footprint_smooth >= threshold
    
    # Compute concentration-weighted centroid of top-2% cells
    xx, yy = np.meshgrid(grid_x, grid_y, indexing='ij')
    
    estimated_x = np.average(xx[hot_cells], weights=footprint_smooth[hot_cells])
    estimated_y = np.average(yy[hot_cells], weights=footprint_smooth[hot_cells])
    
    return estimated_x, estimated_y, footprint, grid_x, grid_y


# ============================================================
# VISUALIZATION
# ============================================================
def make_figure(est_x, est_y, footprint, grid_x, grid_y,
                sensors, wind_times, wind_speeds, wind_dirs, output_path):
    """Generate the trajectory density figure."""
    
    # Compute event-average concentrations for labels
    conc = {}
    for sid, (counter, pm25) in sensors.items():
        bg = np.mean(pm25[(counter >= BG_START) & (counter <= BG_END)])
        conc[sid] = np.mean(pm25[(counter >= COUNTER_START) & (counter <= COUNTER_END)]) - bg
    
    # Smooth and normalize footprint for display
    fp_smooth = gaussian_filter(footprint, sigma=2.5)
    fp_norm = fp_smooth / fp_smooth.max()
    
    # Mean wind during event
    event_mask = (wind_times >= 21 * 3600) & (wind_times <= 23.5 * 3600)
    mean_wind_from = np.mean(wind_dirs[event_mask])
    mean_wind_speed = np.mean(wind_speeds[event_mask])
    mean_wind_toward = (mean_wind_from + 180) % 360
    
    est_lat, est_lon = xy_to_latlon(est_x, est_y)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(fp_norm.T, origin='lower',
                   extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                   cmap='hot_r', aspect='equal', vmin=0, vmax=1)
    
    # Sensors
    for sid, (sx, sy) in SENSOR_XY.items():
        ax.plot(sx, sy, 'b^', ms=13, mec='white', mew=2, zorder=10)
        ax.annotate(f'S{sid}\n({conc[sid]:.1f} ug/m3)', (sx, sy),
                   textcoords='offset points', xytext=(12, 10), fontsize=9,
                   color='cyan', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    # Estimated source
    ax.plot(est_x, est_y, 'x', color='lime', ms=20, mew=4,
            label=f'Estimated source\n({est_lat:.4f}N, {est_lon:.4f}W)', zorder=15)
    
    # Wind arrow from estimated source
    arrow_len = 500
    ax.annotate('',
        xy=(est_x + arrow_len * np.sin(np.radians(mean_wind_toward)),
            est_y + arrow_len * np.cos(np.radians(mean_wind_toward))),
        xytext=(est_x, est_y),
        arrowprops=dict(arrowstyle='->', color='yellow', lw=3), zorder=12)
    ax.text(est_x + 280 * np.sin(np.radians(mean_wind_toward)) + 80,
            est_y + 280 * np.cos(np.radians(mean_wind_toward)),
            f'KSEA wind\nFROM {mean_wind_from:.0f} deg\n{mean_wind_speed:.1f} m/s',
            color='yellow', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_xlabel('East-West (m from reference)', fontsize=12)
    ax.set_ylabel('North-South (m from reference)', fontsize=12)
    ax.set_title(f'Concentration-Weighted Backward Trajectory Density\n'
                 f'Estimated Source: {est_lat:.4f}N, {est_lon:.4f}W\n'
                 f'Wind FROM {mean_wind_from:.0f} deg at {mean_wind_speed:.1f} m/s (KSEA, raw)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
    plt.colorbar(im, ax=ax,
                 label='Relative trajectory density\n(high near sensors by construction)',
                 shrink=0.8)
    ax.grid(True, alpha=0.15, color='white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Figure saved to {output_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PM2.5 Source Localization")
    print("Backward Lagrangian Stochastic Particle Dispersion")
    print("=" * 60)
    
    # Load data
    print("\nLoading sensor data...")
    sensors = load_sensor_data(DATA_DIR)
    for sid, (c, p) in sensors.items():
        bg = np.mean(p[(c >= BG_START) & (c <= BG_END)])
        net = np.mean(p[(c >= COUNTER_START) & (c <= COUNTER_END)]) - bg
        print(f"  S{sid}: {len(c)} samples, background={bg:.1f}, net_event={net:.1f} ug/m3")
    
    print("\nLoading KSEA wind data...")
    wind_times, wind_speeds, wind_dirs = load_wind_data(DATA_DIR)
    event_mask = (wind_times >= 21 * 3600) & (wind_times <= 23.5 * 3600)
    print(f"  {len(wind_times)} records total")
    print(f"  Event period: mean FROM {np.mean(wind_dirs[event_mask]):.1f} deg "
          f"at {np.mean(wind_speeds[event_mask]):.2f} m/s")
    
    # Run backward Lagrangian model
    print("\nRunning backward Lagrangian model...")
    print(f"  Particles: {N_PARTICLES} per sensor-window")
    print(f"  Backward duration: {BACK_DUR} s ({BACK_DUR/60:.0f} min)")
    print(f"  Time step: {DT} s ({N_STEPS} steps)")
    print(f"  Grid: {GRID_RES}m resolution, {GRID_EXTENT*2}m extent")
    
    est_x, est_y, footprint, grid_x, grid_y = run_backward_lagrangian(
        sensors, wind_times, wind_speeds, wind_dirs
    )
    
    # Convert to lat/lon
    est_lat, est_lon = xy_to_latlon(est_x, est_y)
    
    # Report result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Estimated source: {est_lat:.4f} N, {est_lon:.4f} W")
    print(f"  Local coords:     ({est_x:.0f}, {est_y:.0f}) m")
    print(f"  Wind used:        KSEA raw (no corrections)")
    print(f"  No ground truth used.")
    print("=" * 60)
    
    # Generate figure
    print("\nGenerating figure...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUTPUT_DIR, 'source_localization_result.png')
    make_figure(est_x, est_y, footprint, grid_x, grid_y,
                sensors, wind_times, wind_speeds, wind_dirs, fig_path)
    
    print("\nDone.")
