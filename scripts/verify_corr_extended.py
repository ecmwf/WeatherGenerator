#!/usr/bin/env python3
"""
Extended verification of spatial correlation for multiple variables.

Includes near-surface winds, stratospheric winds, humidity, and temperature.
"""

import numpy as np
import anemoi.datasets as anemoi_datasets


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    print("Loading dataset...")
    ds = anemoi_datasets.open_dataset(
        '/capstor/store/cscs/userlab/ch17/data/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr'
    )

    lats = ds.latitudes
    lons = ds.longitudes
    n_points = len(lats)

    # Extended variable set
    variables = [
        # Reference large-scale fields
        'z_500', 't_850', 'tp',
        # Near-surface winds
        '10u', '10v', 'u_1000', 
        # Humidity
        'q_850',
        # 2m temperature
        '2t',
        # Stratospheric winds (50 hPa)
        'u_50', 'v_50',
    ]
    
    var_indices = {}
    for v in variables:
        if v in ds.name_to_index:
            var_indices[v] = ds.name_to_index[v]
        else:
            print(f"Warning: {v} not found in dataset")

    print(f"Grid has {n_points} points")
    print(f"Variables to analyze: {list(var_indices.keys())}")

    # For proper distance sampling on O96 grid
    rng = np.random.default_rng(42)
    n_anchors = 200
    anchors = rng.choice(n_points, size=n_anchors, replace=False)

    bins = [0, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000]

    # Load data for a few timesteps
    t_samples = [1000, 3000, 5000, 10000, 20000]

    # Initialize results storage
    all_results = {v: [] for v in var_indices.keys()}

    for t_idx in t_samples:
        print(f"\nProcessing timestep {t_idx}...")
        sample = ds[t_idx]

        # Load and normalize all variables
        data_norm = {}
        for v, idx in var_indices.items():
            data = sample[idx, 0, :]
            std = data.std()
            if std > 1e-10:
                data_norm[v] = (data - data.mean()) / std
            else:
                data_norm[v] = np.zeros_like(data)

        # Accumulate bin correlations
        bin_corrs = {v: {i: [] for i in range(len(bins) - 1)} for v in var_indices.keys()}

        for anc in anchors:
            # Compute distances from anchor to all points
            dists = haversine_distance_km(lats[anc], lons[anc], lats, lons)

            for b in range(len(bins) - 1):
                mask = (dists >= bins[b]) & (dists < bins[b + 1])
                if mask.sum() > 0:
                    for v in var_indices.keys():
                        bin_corrs[v][b].extend((data_norm[v][anc] * data_norm[v][mask]).tolist())

        # Compute mean for this timestep
        for v in var_indices.keys():
            result = []
            for b in range(len(bins) - 1):
                result.append(np.mean(bin_corrs[v][b]) if bin_corrs[v][b] else np.nan)
            all_results[v].append(result)

    # Print results table
    print("\n" + "=" * 140)
    print("SPATIAL CORRELATION vs DISTANCE (averaged over 5 timesteps)")
    print("=" * 140)
    
    # Header
    header = f"{'Distance (km)':>15}"
    for v in var_indices.keys():
        header += f" | {v:>8}"
    print(header)
    print("-" * 140)

    # Data rows
    for b in range(len(bins) - 1):
        row = f'{bins[b]:>6}-{bins[b + 1]:<6} km'
        for v in var_indices.keys():
            avg = np.nanmean([r[b] for r in all_results[v]])
            row += f" | {avg:>8.4f}"
        print(row)

    # Estimate e-folding distances
    print("\n" + "=" * 80)
    print("E-FOLDING DISTANCES (correlation = 1/e ≈ 0.368)")
    print("=" * 80)
    
    bin_centers = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    threshold = 1 / np.e
    
    l_corr_values = {}
    
    for v in var_indices.keys():
        avg = np.array([np.nanmean([r[b] for r in all_results[v]]) for b in range(len(bins) - 1)])
        # Find first bin below threshold
        below = avg < threshold
        if below.any():
            idx = np.argmax(below)
            if idx > 0:
                # Linear interpolation
                d1, d2 = bin_centers[idx - 1], bin_centers[idx]
                c1, c2 = avg[idx - 1], avg[idx]
                l_corr = d1 + (d2 - d1) * (c1 - threshold) / (c1 - c2 + 1e-10)
                l_corr_values[v] = l_corr
                print(f"  {v:>8}: L_corr ≈ {l_corr:>6.0f} km")
            else:
                l_corr_values[v] = bins[1] / 2
                print(f"  {v:>8}: L_corr < {bins[1]} km (below threshold in first bin)")
        else:
            l_corr_values[v] = bins[-1]
            print(f"  {v:>8}: L_corr > {bins[-1]} km (correlation never drops below threshold)")

    # Summary sorted by correlation length
    print("\n" + "=" * 80)
    print("SUMMARY: Variables sorted by spatial correlation length")
    print("=" * 80)
    sorted_vars = sorted(l_corr_values.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Variable':>12} | {'L_corr (km)':>12} | Physical interpretation")
    print("-" * 80)
    for v, l in sorted_vars:
        if 'z_' in v:
            interp = "Geopotential - large-scale dynamics"
        elif 't_' in v or v == '2t':
            interp = "Temperature"
        elif 'u_' in v or 'v_' in v or '10' in v:
            if '50' in v:
                interp = "Stratospheric wind (50 hPa)"
            elif '1000' in v:
                interp = "Near-surface wind (1000 hPa)"
            elif '10' in v:
                interp = "10m wind"
            else:
                interp = "Wind"
        elif 'q_' in v:
            interp = "Humidity"
        elif v == 'tp':
            interp = "Total precipitation - local/convective"
        else:
            interp = ""
        print(f"{v:>12} | {l:>12.0f} | {interp}")


if __name__ == '__main__':
    main()
