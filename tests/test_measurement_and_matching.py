"""
Unit tests that don't require a trained model or GPU - they exercise
the deterministic geometry (measurement) and matching logic directly,
which is where most bugs are cheap to catch early.

Run with: pytest tests/
"""
import numpy as np
import SimpleITK as sitk
import pytest

from src.measurement.anatomical_measurement import extract_anatomical_measurements
from src.matching.implant_matcher import _score_candidate
from src.database.models import ComponentType, ImplantComponent


def _make_synthetic_label_image():
    """
    Build a tiny synthetic label volume with a known-size 'femur' block
    and 'tibia' block, so we can assert the measurement code recovers
    the expected mm dimensions exactly (a real CT mask will never be
    this clean, but this validates the geometry math is correct).
    """
    shape = (40, 40, 40)  # (z, y, x)
    array = np.zeros(shape, dtype=np.uint8)

    # Femur block: 10 (z) x 12 (y, AP) x 8 (x, ML) voxels, label=1
    array[5:15, 5:17, 5:13] = 1
    # Tibia block: 10 (z) x 10 (y, AP) x 9 (x, ML) voxels, label=2
    array[20:30, 8:18, 6:15] = 2

    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))  # 1mm isotropic -> voxel counts == mm
    return image


def test_extract_anatomical_measurements_recovers_known_dimensions():
    image = _make_synthetic_label_image()
    results = extract_anatomical_measurements(image)

    # ML width = extent along X axis; AP = extent along Y axis.
    # Block spans indices 5:13 (8 voxels) -> max-min = 7 (inclusive extent in coords).
    assert results["femur"].ml_width_mm == pytest.approx(7.0, abs=0.01)
    assert results["femur"].ap_dimension_mm == pytest.approx(11.0, abs=0.01)

    assert results["tibia"].ml_width_mm == pytest.approx(8.0, abs=0.01)
    assert results["tibia"].ap_dimension_mm == pytest.approx(9.0, abs=0.01)


def test_extract_measurements_raises_on_missing_label():
    array = np.zeros((10, 10, 10), dtype=np.uint8)  # only background
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))

    with pytest.raises(ValueError):
        extract_anatomical_measurements(image)


def test_score_candidate_flags_overhang_risk():
    component = ImplantComponent(
        id=1, manufacturer="Test", system_name="TestSys",
        component_type=ComponentType.FEMORAL, size_label="3",
        ml_width_mm=65.0, ap_dimension_mm=60.0, tolerance_mm=1.5,
    )
    # Patient bone is much smaller than this implant -> overhang risk.
    candidate = _score_candidate(patient_ml=55.0, patient_ap=50.0, component=component)

    assert candidate.overhang_risk is True
    assert candidate.within_tolerance is False
    assert candidate.matching_score > 0


def test_score_candidate_within_tolerance_when_close_match():
    component = ImplantComponent(
        id=2, manufacturer="Test", system_name="TestSys",
        component_type=ComponentType.FEMORAL, size_label="3",
        ml_width_mm=65.0, ap_dimension_mm=60.0, tolerance_mm=1.5,
    )
    candidate = _score_candidate(patient_ml=65.2, patient_ap=59.7, component=component)

    assert candidate.within_tolerance is True
    assert candidate.overhang_risk is False
    assert candidate.undercoverage_risk is False
