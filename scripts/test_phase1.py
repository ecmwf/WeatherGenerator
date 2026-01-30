#!/usr/bin/env python3
"""Test script for Phase 1 implementation of variable-specific masking."""

import sys
sys.path.insert(0, "/users/shickman/work/wg_base/agent-variable-specific-masking/WeatherGenerator/src")

from weathergen.datasets.masking import (
    correlation_length_to_hl_mask,
    time_corr_to_block_size,
    _healpix_cell_size_km,
    EARTH_RADIUS_KM,
)

print("=" * 60)
print("Testing Phase 1 implementation")
print("=" * 60)

print(f"\nEARTH_RADIUS_KM = {EARTH_RADIUS_KM}")

print("\n--- HEALPix cell sizes ---")
for hl in range(7):
    size = _healpix_cell_size_km(hl)
    n_cells = 12 * (4**hl)
    print(f"  Level {hl}: {size:7.1f} km (n_cells={n_cells})")

print("\n--- correlation_length_to_hl_mask tests ---")
test_cases = [
    (2000, "z_500 (large scale)"),
    (1000, "t_500"),
    (500, "t_850"),
    (200, "q_700"),
    (100, "precipitation"),
    (50, "very local"),
]

for l_corr, name in test_cases:
    hl = correlation_length_to_hl_mask(l_corr)
    hl_size = _healpix_cell_size_km(hl)
    print(f"  L_corr={l_corr:4d}km ({name:20s}) -> hl_mask={hl} (cell_size={hl_size:.0f}km)")

print("\n--- time_corr_to_block_size tests (6h data) ---")
for t_corr in [6, 12, 24, 48, 72, 120]:
    block = time_corr_to_block_size(t_corr, 6)
    print(f"  T_corr={t_corr:3d}h -> block_size={block}")

print("\n--- Edge case tests ---")
# Test boundary conditions
assert correlation_length_to_hl_mask(10000) == 1, "Should return hl_min for very large L_corr"
assert correlation_length_to_hl_mask(1) == 5, "Should return hl_max for very small L_corr"
assert time_corr_to_block_size(1, 6) == 1, "Should return min_block for small T_corr"
assert time_corr_to_block_size(1000, 6) == 8, "Should return max_block for large T_corr"
print("  All edge cases passed!")

print("\n" + "=" * 60)
print("All Phase 1 tests passed!")
print("=" * 60)
