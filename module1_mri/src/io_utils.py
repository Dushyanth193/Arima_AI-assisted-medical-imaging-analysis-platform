"""
I/O utilities for MRI volumes.

Uses SimpleITK so spacing, origin and direction always travel with the
pixel data - dropping any of these silently breaks every downstream
measurement (volume, thickness, extrusion all depend on real-world spacing).
"""

import SimpleITK as sitk
import numpy as np


def load_mri(path: str) -> dict:
    """Load a .nii / .nii.gz (or any SimpleITK-readable) volume.

    Returns a dict with the numpy array (z, y, x order) plus the
    geometry needed to convert voxels -> millimetres, and the original
    sitk.Image so it can be reused as a reference for saving masks later.
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # numpy order: (z, y, x)

    return {
        "array": array,
        "spacing": image.GetSpacing(),      # (x, y, z) - sitk order
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "sitk_image": image,
    }


def save_mask(mask_array: np.ndarray, reference_image: sitk.Image, path: str) -> None:
    """Save a numpy mask (z, y, x) back out as a NIfTI file, copying the
    geometry from the image it was derived from so it lines up in any viewer."""
    mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_image.CopyInformation(reference_image)
    sitk.WriteImage(mask_image, path)


def load_reference_database(csv_path: str):
    """Reference DB is just a CSV pointing at image paths + metadata + labels.

    Expected columns: subject_id, image_path, age, sex, bmi, oa_label
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = {"subject_id", "image_path", "age", "sex", "bmi", "oa_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference DB CSV is missing columns: {missing}")
    return df
