"""
NIfTI Image Loader
Loads a .nii / .nii.gz file into a SimpleITK Image and extracts metadata.
"""
from __future__ import annotations

from pathlib import Path
import SimpleITK as sitk

from src.io.metadata import ImageMetadata, extract_image_metadata


class NIfTIReadError(Exception):
    """Raised when NIfTI file loading fails."""


def load_nifti_volume(file_path: str | Path) -> tuple[sitk.Image, ImageMetadata]:
    """
    Reads a single NIfTI volume (.nii or .nii.gz) using SimpleITK.
    Returns (sitk_image, ImageMetadata).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise NIfTIReadError(f"NIfTI file does not exist: {file_path}")

    try:
        image = sitk.ReadImage(str(file_path))
    except Exception as e:
        raise NIfTIReadError(f"Failed to read NIfTI file {file_path}: {e}") from e

    meta = extract_image_metadata(image, modality="CT")
    return image, meta
