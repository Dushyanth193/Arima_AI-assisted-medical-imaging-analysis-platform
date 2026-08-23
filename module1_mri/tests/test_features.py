"""
Validates feature math against synthetic shapes with known ground truth -
e.g. a sphere of radius r has volume (4/3)*pi*r^3, so we can check
compute_volume_cm3 against the analytical answer rather than trusting it blindly.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_volume_cm3, compute_thickness_mm, extract_features


def make_sphere_mask(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    return (dist <= radius).astype(np.uint8)


def test_volume_matches_analytical_sphere_volume():
    spacing = (1.0, 1.0, 1.0)  # 1mm isotropic -> voxel volume = 1 mm^3
    radius_voxels = 10
    mask = make_sphere_mask((30, 30, 30), (15, 15, 15), radius_voxels)

    computed_cm3 = compute_volume_cm3(mask, spacing)
    analytical_mm3 = (4/3) * np.pi * (radius_voxels ** 3)
    analytical_cm3 = analytical_mm3 / 1000.0

    pct_error = abs(computed_cm3 - analytical_cm3) / analytical_cm3 * 100
    print(f"  computed={computed_cm3:.3f} cm^3, analytical={analytical_cm3:.3f} cm^3, error={pct_error:.1f}%")
    # voxelized sphere vs. perfect sphere -> allow discretization error
    assert pct_error < 10, "Voxelized sphere volume too far from analytical value"


def test_volume_scales_with_spacing():
    mask = make_sphere_mask((20, 20, 20), (10, 10, 10), 5)
    vol_1mm = compute_volume_cm3(mask, (1.0, 1.0, 1.0))
    vol_2mm = compute_volume_cm3(mask, (2.0, 2.0, 2.0))
    print(f"  1mm spacing -> {vol_1mm:.3f} cm^3, 2mm spacing -> {vol_2mm:.3f} cm^3")
    # doubling spacing in all 3 dims should scale volume by 2^3 = 8x
    assert abs(vol_2mm / vol_1mm - 8.0) < 0.5


def test_thickness_on_uniform_slab():
    # a flat slab of known thickness should report close to that thickness
    slab = np.zeros((5, 40, 40), dtype=np.uint8)
    slab[1:4, 10:30, 10:30] = 1  # 3 voxels thick along axis 0
    spacing = (1.0, 1.0, 1.0)
    thickness = compute_thickness_mm(slab, spacing)
    print(f"  measured thickness={thickness:.2f} mm (slab is 3 voxels thick)")
    assert 1.5 <= thickness <= 4.5


def test_empty_mask_returns_zero():
    empty = np.zeros((10, 10, 10), dtype=np.uint8)
    assert compute_volume_cm3(empty, (1, 1, 1)) == 0.0
    assert compute_thickness_mm(empty, (1, 1, 1)) == 0.0


def test_extract_features_without_tibia_mask():
    mask = make_sphere_mask((20, 20, 20), (10, 10, 10), 5)
    features = extract_features(mask, (1.0, 1.0, 1.0), tibia_mask=None)
    assert features["meniscus_extrusion_mm"] is None
    assert features["meniscus_volume_cm3"] > 0
    print(f"  features={features}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  PASSED\n")
    print(f"All {len(tests)} feature tests passed.")
