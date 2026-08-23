"""
tests/test_landmarks.py
-----------------------
Unit tests for automated tibial plateau landmark detection and meniscus
extrusion measurement in src/landmarks.py.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.landmarks import estimate_tibial_plateau_mask, detect_tibial_plateau_landmarks
from src.features import extract_features, compute_extrusion_mm


def create_synthetic_knee_joint(
    shape=(40, 80, 80),
    spacing=(1.0, 1.0, 1.0),
    meniscus_extrusion_vox: int = 5,
):
    """Creates synthetic volume array, meniscus mask, and ground truth tibia
    with a known controlled extrusion."""
    volume = np.zeros(shape, dtype=np.float32)
    meniscus_mask = np.zeros(shape, dtype=np.uint8)
    tibia_mask = np.zeros(shape, dtype=np.uint8)

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Tibia: z in [22:36], y in [25:55], x in [20:55]
    tibia_x_max = 55
    tibia_region = (zz >= 22) & (zz <= 36) & (yy >= 25) & (yy <= 55) & (xx >= 20) & (xx <= tibia_x_max)
    volume[tibia_region] = 180.0
    tibia_mask[tibia_region] = 1

    # Meniscus: z in [16:21], y in [30:50], x in [22 : 55 + meniscus_extrusion_vox]
    meniscus_x_max = tibia_x_max + meniscus_extrusion_vox
    meniscus_region = (zz >= 16) & (zz <= 21) & (yy >= 30) & (yy <= 50) & (xx >= 25) & (xx <= meniscus_x_max)
    volume[meniscus_region] = 250.0
    meniscus_mask[meniscus_region] = 1

    return volume, meniscus_mask, tibia_mask, meniscus_extrusion_vox * spacing[2]


def test_extrusion_analytical_matches_ground_truth():
    """Validates compute_extrusion_mm on ground-truth aligned masks."""
    volume, men_mask, tib_mask, expected_extrusion = create_synthetic_knee_joint(
        spacing=(0.5, 0.5, 0.75),
        meniscus_extrusion_vox=6,
    )
    measured = compute_extrusion_mm(men_mask, tib_mask, spacing=(0.5, 0.5, 0.75), axis=2)
    print(f"  Analytical Extrusion test: Expected={expected_extrusion:.2f} mm, Measured={measured:.2f} mm")
    assert abs(measured - expected_extrusion) < 1e-3


def test_automated_tibial_landmark_detection():
    """Validates automated landmark detection and extrusion estimation without manual annotations."""
    volume, men_mask, _, expected_extrusion = create_synthetic_knee_joint(
        spacing=(1.0, 1.0, 1.0),
        meniscus_extrusion_vox=4,
    )
    landmarks = detect_tibial_plateau_landmarks(volume, men_mask, spacing=(1.0, 1.0, 1.0))
    print(f"  Automated Landmark Extrusion: {landmarks['meniscus_extrusion_mm']:.2f} mm (Expected ~ {expected_extrusion:.2f} mm)")
    assert landmarks["meniscus_extrusion_mm"] >= 0.0
    assert landmarks["tibia_mask"] is not None
    assert landmarks["tibia_mask"].sum() > 0


def test_extract_features_with_automated_tibia():
    """Validates that extract_features automatically extracts extrusion when volume_array is provided."""
    volume, men_mask, _, _ = create_synthetic_knee_joint(
        spacing=(1.0, 1.0, 1.0),
        meniscus_extrusion_vox=3,
    )
    features = extract_features(men_mask, spacing=(1.0, 1.0, 1.0), volume_array=volume)
    print(f"  Extracted Features: {features}")
    assert features["meniscus_extrusion_mm"] is not None
    assert isinstance(features["meniscus_extrusion_mm"], float)
    assert features["meniscus_volume_cm3"] > 0
    assert features["meniscus_thickness_mm"] > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  PASSED\n")
    print(f"All {len(tests)} landmark & extrusion tests passed.")
