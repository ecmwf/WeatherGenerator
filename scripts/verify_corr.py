#!/usr/bin/env python3
"""
Quick verification of spatial correlation for z_500 vs t_850.

Uses neighbor-aware sampling to properly sample short-distance correlations
on the O96 octahedral reduced Gaussian grid.
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

    # Get variable indices - extended set
    variables = [
        # Reference large-scale fields
        'z_500', 't_850', 'tp',
        # Near-surface winds
        '10u', '10v', 'u_1000', 'v_1000',
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

    # For proper distance sampling on O96 grid:
    # Use a stratified approach: for N anchor points, compute distance to all neighbors
    rng = np.random.default_rng(42)
    n_anchors = 200  # More anchors for better short-distance sampling
    anchors = rng.choice(n_points, size=n_anchors, replace=False)

    bins = [0, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000]

    # Load data for a few timesteps
    t_samples = [1000, 3000, 5000, 10000, 20000]

    all_results = {'z_500': [], 't_850': [], 'tp': []}

    for t_idx in t_samples:
        print(f"\nProcessing timestep {t_idx}...")
        sample = ds[t_idx]

        z500 = sample[z500_idx, 0, :]
        t850 = sample[t850_idx, 0, :]
        tp = sample[tp_idx, 0, :]

        # Normalize
        z500_n = (z500 - z500.mean()) / z500.std()
        t850_n = (t850 - t850.mean()) / t850.std()
        tp_n = (tp - tp.mean()) / tp.std()

        # Accumulate bin correlations
        bin_corrs_z = {i: [] for i in range(len(bins) - 1)}
        bin_corrs_t = {i: [] for i in range(len(bins) - 1)}
        bin_corrs_p = {i: [] for i in range(len(bins) - 1)}

        for anc in anchors:
            # Compute distances from anchor to all points
            dists = haversine_distance_km(lats[anc], lons[anc], lats, lons)

            for b in range(len(bins) - 1):
                mask = (dists >= bins[b]) & (dists < bins[b + 1])
                if mask.sum() > 0:
                    bin_corrs_z[b].extend((z500_n[anc] * z500_n[mask]).tolist())
                    bin_corrs_t[b].extend((t850_n[anc] * t850_n[mask]).tolist())
                    bin_corrs_p[b].extend((tp_n[anc] * tp_n[mask]).tolist())

        # Compute mean for this timestep
        z_result = []
        t_result = []
        p_result = []
        for b in range(len(bins) - 1):
            z_result.append(np.mean(bin_corrs_z[b]) if bin_corrs_z[b] else np.nan)
            t_result.append(np.mean(bin_corrs_t[b]) if bin_corrs_t[b] else np.nan)
            p_result.append(np.mean(bin_corrs_p[b]) if bin_corrs_p[b] else np.nan)

        all_results['z_500'].append(z_result)
        all_results['t_850'].append(t_result)
        all_results['tp'].append(p_result)

    # Print results
    print("\n" + "=" * 70)
    print("SPATIAL CORRELATION vs DISTANCE (averaged over 5 timesteps)")
    print("=" * 70)
    print(f"{'Distance (km)':>15} | {'z_500':>10} | {'t_850':>10} | {'tp':>10} | n_pairs")
    print("-" * 70)

    for b in range(len(bins) - 1):
        z_avg = np.nanmean([r[b] for r in all_results['z_500']])
        t_avg = np.nanmean([r[b] for r in all_results['t_850']])
        p_avg = np.nanmean([r[b] for r in all_results['tp']])
        # Get pair count from last timestep's accumulation
        n_pairs = len(bin_corrs_z[b])
        print(f'{bins[b]:>6}-{bins[b + 1]:<6} km | {z_avg:>10.4f} | {t_avg:>10.4f} | {p_avg:>10.4f} | {n_pairs}')

    # Estimate e-folding distances
    print("\n" + "=" * 70)
    print("E-FOLDING DISTANCES (correlation = 1/e ≈ 0.368)")
    print("=" * 70)
    bin_centers = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    threshold = 1 / np.e

    for name in ['z_500', 't_850', 'tp']:
        avg = np.array([np.nanmean([r[b] for r in all_results[name]]) for b in range(len(bins) - 1)])
        # Find first bin below threshold
        below = avg < threshold
        if below.any():
            idx = np.argmax(below)
            if idx > 0:
                # Linear interpolation
                d1, d2 = bin_centers[idx - 1], bin_centers[idx]
                c1, c2 = avg[idx - 1], avg[idx]
                l_corr = d1 + (d2 - d1) * (c1 - threshold) / (c1 - c2 + 1e-10)
                print(f"  {name}: L_corr ≈ {l_corr:.0f} km")
            else:
                print(f"  {name}: L_corr < {bins[1]} km (below threshold in first bin)")
        else:
            print(f"  {name}: L_corr > {bins[-1]} km (correlation never drops below threshold)")

    # Summary
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    z_avg = np.array([np.nanmean([r[b] for r in all_results['z_500']]) for b in range(len(bins) - 1)])
    t_avg = np.array([np.nanmean([r[b] for r in all_results['t_850']]) for b in range(len(bins) - 1)])
    p_avg = np.array([np.nanmean([r[b] for r in all_results['tp']]) for b in range(len(bins) - 1)])

    print("\nAt 1000 km distance:")
    b = 2  # 500-1000 km bin
    print(f"  z_500: {z_avg[b]:.3f}, t_850: {t_avg[b]:.3f}, tp: {p_avg[b]:.3f}")
    if z_avg[b] > t_avg[b]:
        print("  → z_500 has LONGER correlation length than t_850 ✓")
    else:
        print("  → t_850 has LONGER correlation length than z_500 (unexpected?)")

    print("\nAt 3000 km distance:")
    b = 5  # 2000-3000 km bin
    print(f"  z_500: {z_avg[b]:.3f}, t_850: {t_avg[b]:.3f}, tp: {p_avg[b]:.3f}")

    print("\nPrecipitation (tp) should have much shorter correlation scale:")
    if p_avg[2] < z_avg[2] and p_avg[2] < t_avg[2]:
        print("  → tp correlation drops faster than z_500 and t_850 ✓")


if __name__ == '__main__':
    main()
