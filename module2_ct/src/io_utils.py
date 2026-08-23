"""
io_utils.py
-----------
I/O utilities for Module 2: CT bone segmentation and resection analysis.
Handles loading CT scans (.nii, .nii.gz, DICOM), extracting Hounsfield units,
and saving segmented bone masks.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import SimpleITK as sitk


def load_ct(file_path: str) -> Dict[str, Any]:
    """Loads a 3D CT scan from a NIfTI file or directory of DICOM slices.

    Parameters
    ----------
    file_path : str
        Path to .nii or .nii.gz file, or directory with DICOM series.

    Returns
    -------
    dict with keys:
        'array' : np.ndarray, float32 array in (Z, Y, X) order
        'spacing' : tuple of floats (dx, dy, dz) in mm (X, Y, Z order from ITK)
        'origin' : tuple of floats (ox, oy, oz)
        'direction' : tuple of floats (matrix elements)
        'sitk_image' : sitk.Image object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CT file not found: {file_path}")

    if os.path.isdir(file_path):
        # Load DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_path)
        if not dicom_names:
            raise ValueError(f"No DICOM series found in directory: {file_path}")
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    else:
        # Load NIfTI file
        img = sitk.ReadImage(file_path)

    # Convert to float32 Hounsfield Units (HU)
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(img_f)  # Shape: (Z, Y, X)

    return {
        "array": array,
        "spacing": img.GetSpacing(),  # (dx, dy, dz) in mm
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "sitk_image": img_f,
    }


def save_ct_mask(
    mask_array: np.ndarray,
    reference_image: sitk.Image,
    output_path: str,
) -> None:
    """Saves a binary or multi-label bone mask array as a NIfTI (.nii.gz) file,
    copying geometry metadata from a reference SimpleITK image.

    Parameters
    ----------
    mask_array : np.ndarray
        Multi-label mask array in (Z, Y, X) order (e.g. 1=Femur, 2=Tibia).
    reference_image : sitk.Image
        Reference CT image for spatial origin, spacing, and direction cosines.
    output_path : str
        Destination path (.nii or .nii.gz).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mask_itk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_itk.CopyInformation(reference_image)
    sitk.WriteImage(mask_itk, output_path)


def save_ct_volume(
    volume_array: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Saves a 3D CT volume array as a NIfTI file with specified physical spacing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img = sitk.GetImageFromArray(volume_array.astype(np.float32))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, output_path)
