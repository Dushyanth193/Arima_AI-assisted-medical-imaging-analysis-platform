"""
Unit tests for Preprocessing & Loader modules.
"""
import pytest
import SimpleITK as sitk
import numpy as np

from src.io.metadata import extract_image_metadata
from src.preprocessing.ct_preprocessing import preprocess_ct, load_ct_volume


def test_extract_metadata():
    img = sitk.Image([10, 20, 30], sitk.sitkFloat32)
    img.SetSpacing([0.5, 0.5, 1.0])
    img.SetOrigin([0.0, 0.0, 0.0])

    meta = extract_image_metadata(img, modality="CT")
    assert meta.dimensions == (10, 20, 30)
    assert meta.spacing == (0.5, 0.5, 1.0)
    assert meta.origin == (0.0, 0.0, 0.0)
    assert meta.modality == "CT"


def test_preprocess_ct_synthetic(tmp_path):
    # Create a synthetic 3D CT volume as NIfTI
    arr = np.random.randint(-1000, 1000, size=(40, 50, 60), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([0.8, 0.8, 1.2])
    nii_path = tmp_path / "test_ct.nii.gz"
    sitk.WriteImage(img, str(nii_path))

    processed = preprocess_ct(nii_path)
    assert isinstance(processed, sitk.Image)
    assert processed.GetSpacing() == (1.0, 1.0, 1.0)
