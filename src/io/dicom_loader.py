"""
DICOM Image Loader
Loads a DICOM series directory into a SimpleITK Image and extracts metadata.
"""
from __future__ import annotations

from pathlib import Path
import SimpleITK as sitk

from src.io.metadata import ImageMetadata, extract_image_metadata


class DICOMReadError(Exception):
    """Raised when DICOM directory reading fails."""


def load_dicom_series(dicom_dir: str | Path) -> tuple[sitk.Image, ImageMetadata]:
    """
    Reads a DICOM series from a directory using SimpleITK GDCM reader.
    Returns (sitk_image, ImageMetadata).
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise DICOMReadError(f"Directory does not exist: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise DICOMReadError(f"No DICOM series found in directory: {dicom_dir}")

    # Select the largest series by file count
    best_series, max_files = None, -1
    for s_id in series_ids:
        file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), s_id)
        if len(file_names) > max_files:
            max_files = len(file_names)
            best_series = file_names

    if not best_series:
        raise DICOMReadError(f"No valid DICOM files found in {dicom_dir}")

    reader.SetFileNames(best_series)
    image = reader.Execute()
    meta = extract_image_metadata(image, modality="CT")
    return image, meta
