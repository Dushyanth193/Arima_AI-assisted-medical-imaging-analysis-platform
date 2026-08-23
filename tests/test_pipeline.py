"""
Integration test for full inference pipeline.
"""
import pytest
import numpy as np
import SimpleITK as sitk

from src.pipeline.inference_pipeline import run_pipeline_for_file
from src.database.models import init_db


def test_full_pipeline_synthetic(tmp_path):
    init_db()

    # Create a synthetic 3D scan volume
    arr = np.zeros((40, 60, 60), dtype=np.int16)
    # Add synthetic bone structures
    arr[5:20, 15:45, 15:45] = 1200  # Femur-like region
    arr[22:38, 15:45, 15:45] = 1000  # Tibia-like region

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    nii_path = tmp_path / "synthetic_patient.nii.gz"
    sitk.WriteImage(img, str(nii_path))

    res = run_pipeline_for_file(
        input_path=nii_path,
        patient_reference="test-synthetic-001",
    )

    assert res["patient_reference"] == "test-synthetic-001"
    assert "measurements" in res
    assert "femoral" in res
    assert "tibial" in res
    assert "mesh_paths" in res
