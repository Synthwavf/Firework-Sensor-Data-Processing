#!/usr/bin/env python3
"""
Export v3 — Convert grid simulation to point-based heatmap data
for heatmap.js + Leaflet TimeDimension (matching the folium reference style).

Output format per timestep: [[lat, lng, value], [lat, lng, value], ...]
Only includes cells with nonzero concentration (sparse).
"""

import numpy as np
import csv
import os
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR

SOURCES = {
    # Source locations from bLS Hanna-D z=200 m (sensitivity-sweep point estimate,
    # 189 m from known barge; see source_location_estimation_*.py).
    'pm25': {'lat': 47.6387, 'lon': -122.3358, 'label': 'PM2.5', 'csv_col': 17},
    'pm1':  {'lat': 47.6387, 'lon': -122.3358, 'label': 'PM1',   'csv_col': 16},
}

SENSOR_LOCS = {
    1: (47.6378, -122.3295, '2031 Fairview Ave E'),
    2: (47.6469, -122.3263, '2838 Fairview Ave E'),
    3: (47.6493, -122.3316, '2199 N Northlake Way'),
    5: (47.6299, -122.3394, '1200 Westlake Ave N'),
    6: (47.6267, -122.3353, '809 Fairview Pl N'),
}

# Turbulence parameters: Hanna 1982 NEUTRAL at z=200 m (2D horizontal formulation).
# See source_location_estimation_pm25.py header comment for full derivation chain.
SIGMA_H = 0.33        # horizontal turbulent velocity std (m/s)
T_LH    = 148         # horizontal Lagrangian integral timescale (s)
TAU_RES = 500         # residence-time decay (s) -- replaces old z<60 m filter
DT      = 6
RANDOM_SEED = 42

EMISSION_START = 22 * 3600 + 20 * 60
EMISSION_END   = 22 * 3600 + 40 * 60
EMISSION_INTERVAL = 30
N_PARTICLES_PER_PUFF = 1500
# Burst dispersion (2D): initial radial expansion of each puff's particles
# to represent the sub-DT spread of a mortar's fireball before the Langevin
# turbulence takes over.
BURST_RADIAL_SPEED_MEAN = 8.0
BURST_RADIAL_SPEED_STD  = 4.0

REF_LAT = 47.638
REF_LON = -122.333
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111132.0 * np.cos(np.radians(REF_LAT))

ALL_LATS = [v[0] for v in SENSOR_LOCS.values()] + [47.6391, 47.6389]
ALL_LONS = [v[1] for v in SENSOR_LOCS.values()] + [-122.3352, -122.3352]
CENTROID_LAT = np.mean(ALL_LATS)
CENTROID_LON = np.mean(ALL_LONS)

# Grid for particle binning — moderate resolution
GRID_RES = 100       # meters (coarser — heatmap.js does its own radius smoothing)
GRID_EXTENT = 10000  # 10 km from center — plume will naturally die before reaching this

SIM_START = 22 * 3600 + 15 * 60
SIM_END   = 23 * 3600 + 20 * 60
SNAPSHOT_INTERVAL = 60

COUNTER_DT = 1.111
T0_SECONDS = 19 * 3600 + 59 * 60 + 30
BG_START, BG_END = 1000, 5000


def latlon_to_xy(lat, lon):
    return (lon - REF_LON) * M_PER_DEG_LON, (lat - REF_LAT) * M_PER_DEG_LAT

def xy_to_latlon(x, y):
    return REF_LAT + y / M_PER_DEG_LAT, REF_LON + x / M_PER_DEG_LON


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
        sensors[sid] = (ts, np.maximum(va - bg, 0))
    return sensors


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


def run_simulation(src_lat, src_lon, is_pm1, wt, ws, wd):
    """Run forward dispersion and return point-based heatmap data."""
    np.random.seed(RANDOM_SEED)
    src_x, src_y = latlon_to_xy(src_lat, src_lon)
    cent_x, cent_y = latlon_to_xy(CENTROID_LAT, CENTROID_LON)

    grid_x = np.arange(cent_x - GRID_EXTENT, cent_x + GRID_EXTENT, GRID_RES)
    grid_y = np.arange(cent_y - GRID_EXTENT, cent_y + GRID_EXTENT, GRID_RES)
    nx, ny = len(grid_x), len(grid_y)

    R_h = np.exp(-DT / T_LH)

    puff_times = np.arange(EMISSION_START, EMISSION_END, EMISSION_INTERVAL)
    n_puffs = len(puff_times)
    max_p = n_puffs * N_PARTICLES_PER_PUFF

    ax = np.full(max_p, src_x); ay = np.full(max_p, src_y)
    aut = np.zeros(max_p); avt = np.zeros(max_p)
    active = np.zeros(max_p, dtype=bool)
    rel_t = np.zeros(max_p)

    for i, pt in enumerate(puff_times):
        i0, i1 = i * N_PARTICLES_PER_PUFF, (i+1) * N_PARTICLES_PER_PUFF
        n = N_PARTICLES_PER_PUFF
        # 2D burst spread: each particle gets a horizontal kick in a random
        # azimuth direction, representing the sub-DT fireball expansion.
        radial_speed = np.abs(np.random.normal(BURST_RADIAL_SPEED_MEAN, BURST_RADIAL_SPEED_STD, n))
        azimuth = np.random.uniform(0, 2 * np.pi, n)
        aut[i0:i1] = radial_speed * np.cos(azimuth)
        avt[i0:i1] = radial_speed * np.sin(azimuth)
        rel_t[i0:i1] = pt

    snapshot_times = list(range(SIM_START, SIM_END + 1, SNAPSHOT_INTERVAL))
    # Output: list of lists of [lat, lng, value]
    all_frames = []
    time_labels = []

    print(f"  Grid: {nx}x{ny}, {n_puffs} puffs, {max_p} particles")

    t = SIM_START
    while t <= SIM_END + 60:
        active |= (rel_t <= t)
        na = np.sum(active)
        u_mean, v_mean = interpolate_wind(t, wt, ws, wd)

        if na > 0:
            nh1 = np.random.randn(na); nh2 = np.random.randn(na)
            aut[active] = R_h * aut[active] + np.sqrt(1 - R_h**2) * SIGMA_H * nh1
            avt[active] = R_h * avt[active] + np.sqrt(1 - R_h**2) * SIGMA_H * nh2
            ax[active] += (u_mean + aut[active]) * DT
            ay[active] += (v_mean + avt[active]) * DT

        for st in snapshot_times:
            if abs(t - st) < DT / 2 and st not in [f[0] for f in all_frames]:
                # Bin particles onto grid with residence-time decay
                # (replaces old z<60 m near-surface filter).
                conc = np.zeros((nx, ny), dtype=np.float64)
                if na > 0:
                    ages = st - rel_t[active]
                    weights = np.exp(-np.maximum(ages, 0.0) / TAU_RES)
                    xi = ((ax[active] - grid_x[0]) / GRID_RES).astype(int)
                    yi = ((ay[active] - grid_y[0]) / GRID_RES).astype(int)
                    v = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                    if np.any(v):
                        np.add.at(conc, (xi[v], yi[v]), weights[v])

                # Smooth
                conc_s = gaussian_filter(conc, sigma=1.5)

                # Apply radial distance-from-source soft fade envelope
                # This ensures plume fades smoothly regardless of grid edges
                FADE_START = 4000   # meters — full intensity within this radius
                FADE_END   = 9000   # meters — zero intensity beyond this radius
                for ri in range(nx):
                    for ci in range(ny):
                        dx = grid_x[ri] - src_x
                        dy = grid_y[ci] - src_y
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > FADE_START:
                            if dist >= FADE_END:
                                conc_s[ri, ci] = 0.0
                            else:
                                # Smooth cosine fade
                                fade = 0.5 * (1 + np.cos(np.pi * (dist - FADE_START) / (FADE_END - FADE_START)))
                                conc_s[ri, ci] *= fade

                # Convert nonzero cells to [lat, lng, value] points
                points = []
                threshold = 0.02 * conc_s.max() if conc_s.max() > 0 else 0
                for ri in range(nx):
                    for ci in range(ny):
                        val = conc_s[ri, ci]
                        if val > threshold:
                            lat, lon = xy_to_latlon(grid_x[ri], grid_y[ci])
                            points.append([round(lat, 6), round(lon, 6), round(float(val), 4)])

                all_frames.append((st, points))
                h, m = divmod(st // 60, 60)
                time_labels.append(f"2025-07-04 {h:02d}:{m:02d}")
                print(f"    {h:02d}:{m:02d} — {len(points)} points, max={conc_s.max():.2f}")
        t += DT

    return all_frames, time_labels


def export_for_html(pm_type, all_frames, time_labels, sensor_data, src_lat, src_lon):
    """Export as a JS-embeddable data structure."""
    # Build the nested array: hm_data[timestep_index] = [[lat, lng, val], ...]
    hm_data = [f[1] for f in all_frames]

    # Sensor values per timestep
    sensor_ts = {}
    for sid, (t_arr, pm_arr) in sensor_data.items():
        vals = []
        for st, _ in all_frames:
            mask = (t_arr >= st - 30) & (t_arr <= st + 30)
            vals.append(round(float(np.mean(pm_arr[mask])), 1) if np.any(mask) else 0.0)
        sensor_ts[str(sid)] = vals

    # Compute global max for the colorbar
    global_max = 0
    for pts in hm_data:
        for p in pts:
            if p[2] > global_max:
                global_max = p[2]

    out = {
        'pm_type': pm_type,
        'source': {'lat': src_lat, 'lon': src_lon},
        'sensors': {str(sid): {'lat': lat, 'lon': lon, 'address': addr}
                    for sid, (lat, lon, addr) in SENSOR_LOCS.items()},
        'time_labels': time_labels,
        'hm_data': hm_data,
        'sensor_ts': sensor_ts,
        'global_max': round(global_max, 2),
    }

    path = os.path.join(OUTPUT_DIR, f'heatmap_v3_{pm_type}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Exported {path} ({size_mb:.1f} MB, {len(hm_data)} frames)")
    return path


if __name__ == '__main__':
    print("Loading wind data...")
    wt, ws, wd = load_wind_data()

    for pm_type, cfg in SOURCES.items():
        print(f"\n{'='*50}")
        print(f"{cfg['label']} — point-based heatmap export")
        sensor_data = load_sensor_data(cfg['csv_col'])
        all_frames, time_labels = run_simulation(
            cfg['lat'], cfg['lon'], pm_type == 'pm1', wt, ws, wd
        )
        export_for_html(pm_type, all_frames, time_labels, sensor_data, cfg['lat'], cfg['lon'])

    print("\nDone.")
