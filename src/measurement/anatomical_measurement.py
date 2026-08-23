"""
Anatomical Feature Extraction
==============================
Implements the "Anatomical Feature Extraction" box of the flow diagram:
    Femur: femoral width, femoral AP dimension
    Tibia: tibial width, tibial AP dimension

This is deliberately NOT a deep-learning step. Once the segmentation
mask exists, these are geometric measurements computed with
scikit-image (region properties, boundary extraction) and SciPy/NumPy
(distance calculations along anatomically-aligned axes) - matching the
"Anatomical measurement: SciPy + NumPy" row of the tech stack table.

Definitions used (consistent with standard TKA pre-op planning measurements):
    - Mediolateral (ML) width  : widest transverse dimension of the distal
                                  femoral condyles / proximal tibial plateau.
    - Anteroposterior (AP) dim : the perpendicular depth from the anterior
                                  to posterior cortical boundary at the
                                  same axial level as the ML measurement.

IMPORTANT LIMITATION (documented, not hidden): these are voxel-grid
"bounding" measurements along principal anatomical axes, not full
surgical-planning landmark-based measurements (e.g. epicondylar axis,
posterior condylar axis) used by commercial planning software. That
level of landmark detection would need its own validated model/algorithm
and is flagged in Limitations & Solutions rather than silently assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure

from src.utils.config import LABELS


@dataclass
class BoneMeasurement:
    label_name: str
    ml_width_mm: float          # mediolateral width
    ap_dimension_mm: float      # anteroposterior dimension
    volume_mm3: float
    centroid_voxel: tuple


def _largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Segmentation models occasionally produce small spurious islands of
    predicted foreground. Keeping only the largest connected component
    per bone is a standard, cheap post-processing cleanup step
    (matches the "Mask processing: clean masks" row of the tech stack).
    """
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def _measure_bone(mask_array: np.ndarray, spacing_mm: tuple, label_value: int, label_name: str) -> BoneMeasurement:
    binary = mask_array == label_value
    binary = _largest_connected_component(binary)

    if not binary.any():
        raise ValueError(f"No voxels found for label '{label_name}' - check segmentation output.")

    coords = np.argwhere(binary)  # (N, 3) in (Z, Y, X) voxel order

    # spacing_mm is (x, y, z) from SimpleITK; mask_array is (z, y, x) from GetArrayFromImage.
    sx, sy, sz = spacing_mm
    coords_mm = coords * np.array([sz, sy, sx])  # convert voxel indices -> physical mm, matching (z,y,x)

    # ML width: extent along the X (left-right / mediolateral) axis.
    ml_width_mm = coords_mm[:, 2].max() - coords_mm[:, 2].min()

    # AP dimension: extent along the Y (anterior-posterior) axis.
    # NOTE: this assumes the volume has already been standardized to a
    # consistent LPS-style orientation (see ct_preprocessing.standardize_orientation),
    # so that image Y corresponds to anatomical anterior-posterior.
    ap_dimension_mm = coords_mm[:, 1].max() - coords_mm[:, 1].min()

    voxel_volume_mm3 = sx * sy * sz
    volume_mm3 = float(binary.sum() * voxel_volume_mm3)

    centroid_voxel = tuple(coords.mean(axis=0))

    return BoneMeasurement(
        label_name=label_name,
        ml_width_mm=float(ml_width_mm),
        ap_dimension_mm=float(ap_dimension_mm),
        volume_mm3=volume_mm3,
        centroid_voxel=centroid_voxel,
    )


def extract_anatomical_measurements(label_image: sitk.Image) -> dict:
    """
    Parameters
    ----------
    label_image : sitk.Image
        Integer label mask with values matching src.utils.config.LABELS
        (0=background, 1=femur, 2=tibia).

    Returns
    -------
    dict with "femur" and "tibia" keys, each a BoneMeasurement.
    """
    mask_array = sitk.GetArrayFromImage(label_image)
    spacing_mm = label_image.GetSpacing()  # (x, y, z)

    results = {}
    for name in ("femur", "tibia"):
        results[name] = _measure_bone(mask_array, spacing_mm, LABELS[name], name)

    return results


def measurements_to_dict(measurements: dict) -> dict:
    """Flatten BoneMeasurement objects into a plain dict for API/DB use."""
    flat = {}
    for bone_name, m in measurements.items():
        flat[f"{bone_name}_ml_width_mm"] = round(m.ml_width_mm, 2)
        flat[f"{bone_name}_ap_dimension_mm"] = round(m.ap_dimension_mm, 2)
        flat[f"{bone_name}_volume_mm3"] = round(m.volume_mm3, 1)
    return flat


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract femur/tibia measurements from a label mask.")
    parser.add_argument("label_mask_path", type=str)
    args = parser.parse_args()

    img = sitk.ReadImage(args.label_mask_path)
    results = extract_anatomical_measurements(img)
    print(json.dumps(measurements_to_dict(results), indent=2))
