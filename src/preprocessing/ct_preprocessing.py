"""
CT Preprocessing
=================
Implements the "CT Preprocessing" box of the flow diagram:
    - Image quality control
    - Noise reduction
    - Standardization
    - Spatial orientation
    - Voxel-size normalization

Input : path to a DICOM series directory OR a single NIfTI file (.nii/.nii.gz)
Output: a standardized, resampled, intensity-normalized SimpleITK.Image,
        ready to be handed to the segmentation dataloader.

Only SimpleITK + NumPy are used here, per the specified tech stack.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from src.utils.config import TARGET_SPACING, HU_CLIP_MIN, HU_CLIP_MAX

logger = logging.getLogger(__name__)


class CTPreprocessingError(Exception):
    """Raised when an input CT volume fails quality control checks."""


def load_ct_volume(input_path: Union[str, Path]) -> sitk.Image:
    """
    Load a CT volume from either a DICOM series directory or a single
    NIfTI file.

    Parameters
    ----------
    input_path : str | Path
        Directory containing a DICOM series, or a path to a .nii/.nii.gz file.

    Returns
    -------
    sitk.Image
        The loaded volume with original spacing/orientation/HU values.
    """
    import zipfile

    input_path = Path(input_path)

    # Handle zip files automatically
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = input_path.parent / f"extracted_{input_path.stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        input_path = extract_dir

    if input_path.is_dir():
        reader = sitk.ImageSeriesReader()
        target_dir = input_path
        series_ids = reader.GetGDCMSeriesIDs(str(target_dir))
        
        if not series_ids:
            # Check subdirectories recursively for DICOM series
            for sub_dir in input_path.rglob("*"):
                if sub_dir.is_dir():
                    sids = reader.GetGDCMSeriesIDs(str(sub_dir))
                    if sids:
                        target_dir = sub_dir
                        series_ids = sids
                        break

        if not series_ids:
            # Check if there are NIfTI files inside directory
            nii_files = list(input_path.rglob("*.nii")) + list(input_path.rglob("*.nii.gz"))
            if nii_files:
                image = sitk.ReadImage(str(nii_files[0]))
                logger.info("Loaded NIfTI volume from directory size=%s spacing=%s", image.GetSize(), image.GetSpacing())
                return image
            raise CTPreprocessingError(f"No DICOM series or NIfTI volume found in {input_path}")

        # If multiple series exist (e.g. scout + diagnostic), take the largest one
        best_series, best_len = None, -1
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(target_dir), sid)
            if len(files) > best_len:
                best_series, best_len = sid, len(files)
        file_names = reader.GetGDCMSeriesFileNames(str(target_dir), best_series)
        reader.SetFileNames(file_names)
        image = reader.Execute()
    elif input_path.suffix.lower() in (".nii", ".gz") or input_path.name.lower().endswith(".nii.gz"):
        image = sitk.ReadImage(str(input_path))
    else:
        raise CTPreprocessingError(f"Unsupported input type: {input_path}")


    logger.info(
        "Loaded volume size=%s spacing=%s origin=%s",
        image.GetSize(), image.GetSpacing(), image.GetOrigin(),
    )
    return image


def quality_control(image: sitk.Image) -> None:
    """
    Basic automated QC gate before the volume enters the pipeline.
    Raises CTPreprocessingError on failure instead of silently
    proceeding with a bad volume (important for a clinical-adjacent tool).
    """
    size = image.GetSize()
    spacing = image.GetSpacing()

    if any(s <= 1 for s in size):
        raise CTPreprocessingError(f"Degenerate volume size: {size}")

    # A knee CT covering the joint should have a reasonable number of
    # axial slices; too few suggests a scout/localizer, not a full scan.
    if size[2] < 5:
        raise CTPreprocessingError(
            f"Volume has only {size[2]} slices - degenerate slice count."
        )


    if any(s <= 0 for s in spacing):
        raise CTPreprocessingError(f"Invalid voxel spacing: {spacing}")

    arr = sitk.GetArrayFromImage(image)
    if np.isnan(arr).any():
        raise CTPreprocessingError("Volume contains NaN voxel values.")

    # HU sanity check: real CT should span a wide range (air ~-1000, bone >300).
    if arr.max() - arr.min() < 200:
        raise CTPreprocessingError(
            "Voxel intensity range is implausibly small for a CT scan; "
            "check that this is CT (not a pre-windowed image export)."
        )


def standardize_orientation(image: sitk.Image, target_orientation: str = "LPS") -> sitk.Image:
    """Reorient the volume to a fixed, consistent anatomical orientation."""
    return sitk.DICOMOrient(image, target_orientation)


import os
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(min(32, os.cpu_count() or 4))


def denoise(image: sitk.Image) -> sitk.Image:
    """
    Mild edge-preserving denoising using 1-iteration curvature flow.
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.CurvatureFlow(image1=image, timeStep=0.125, numberOfIterations=1)



def resample_to_target_spacing(
    image: sitk.Image,
    target_spacing: tuple = TARGET_SPACING,
    is_label: bool = False,
) -> sitk.Image:
    """
    Resample to isotropic target spacing. Uses B-spline interpolation for
    intensity images and nearest-neighbor for label masks (to avoid
    inventing fractional class labels at boundaries).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000 if not is_label else 0)  # air HU, or background label
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)


    return resampler.Execute(image)


def normalize_intensity(image: sitk.Image) -> sitk.Image:
    """
    Clip to a bone-relevant HU window, then rescale to [0, 1].
    This matches the fixed-window normalization approach nnU-Net uses
    for CT (clip to a percentile/anatomically-relevant range, then
    z-score or min-max scale) rather than MRI-style per-scan
    normalization, since CT HU values are physically calibrated and
    comparable across scans.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, HU_CLIP_MIN, HU_CLIP_MAX)
    arr = (arr - HU_CLIP_MIN) / (HU_CLIP_MAX - HU_CLIP_MIN)  # -> [0, 1]

    normalized = sitk.GetImageFromArray(arr)
    normalized.CopyInformation(image)
    return normalized


def preprocess_ct(input_path: Union[str, Path]) -> sitk.Image:
    """
    End-to-end preprocessing pipeline matching the flow diagram's
    "CT Preprocessing" stage:
        load -> quality control -> standardize orientation -> resample
        to isotropic spacing -> denoise -> normalize intensity

    Returns a preprocessed sitk.Image ready for segmentation inference
    or for saving as a training example.
    """
    image = load_ct_volume(input_path)
    quality_control(image)
    image = standardize_orientation(image)
    image = resample_to_target_spacing(image, is_label=False)
    image = denoise(image)
    image = normalize_intensity(image)
    return image



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a knee CT volume.")
    parser.add_argument("input_path", type=str, help="DICOM directory or NIfTI file")
    parser.add_argument("output_path", type=str, help="Output .nii.gz path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_image = preprocess_ct(args.input_path)
    sitk.WriteImage(out_image, args.output_path)
    logger.info("Wrote preprocessed volume to %s", args.output_path)
