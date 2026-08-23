"""
Geometric feature extraction from a meniscus segmentation mask.

All functions take voxel spacing explicitly - never assume isotropic 1mm
voxels, or measurements will be wrong on real clinical data.
"""

import numpy as np
from scipy import ndimage
from skimage import morphology


def compute_volume_cm3(mask: np.ndarray, spacing: tuple) -> float:
    """Voxel count x voxel volume, converted from mm^3 to cm^3."""
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_count = int(np.sum(mask > 0))
    volume_mm3 = voxel_count * voxel_volume_mm3
    return volume_mm3 / 1000.0


def compute_thickness_mm(mask: np.ndarray, spacing: tuple) -> float:
    """Mean thickness via distance transform along the mask's skeleton.

    The distance transform gives, at every foreground voxel, the distance
    to the nearest background voxel (i.e. a local radius). Doubling the
    radius at skeleton points gives a full local thickness estimate.
    """
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0.0

    distance_map = ndimage.distance_transform_edt(mask_bool, sampling=spacing)
    skeleton = morphology.skeletonize(mask_bool)

    if skeleton.sum() == 0:
        # fallback: mean radius x 2 over the whole mask
        return float(distance_map[mask_bool].mean() * 2)

    thickness_values = distance_map[skeleton] * 2
    return float(np.mean(thickness_values))


def compute_extrusion_mm(
    meniscus_mask: np.ndarray,
    tibia_mask: np.ndarray,
    spacing: tuple,
    axis: int = 2,
) -> float:
    """Approximate meniscus extrusion beyond the tibial plateau edge.

    Measures how far the meniscus's extent along `axis` protrudes past the
    tibia's extent along the same axis. `axis` should be the medial-lateral
    direction in your array's coordinate order (commonly axis=2 for a
    (z, y, x) array where x is medial-lateral) - confirm against your
    actual scan orientation before trusting this number clinically.

    Returns 0.0 if either mask is empty or there's no measurable extrusion.
    """
    if meniscus_mask.sum() == 0 or tibia_mask.sum() == 0:
        return 0.0

    men_coords = np.where(meniscus_mask)[axis]
    tib_coords = np.where(tibia_mask)[axis]

    extrusion_voxels = max(0, int(men_coords.max()) - int(tib_coords.max()))
    return float(extrusion_voxels * spacing[axis])


def extract_features(
    meniscus_mask: np.ndarray,
    spacing: tuple,
    tibia_mask: np.ndarray = None,
    volume_array: np.ndarray = None,
) -> dict:
    """Bundles all three meniscus features into one dict, matching the
    'New-Patient Meniscus Features' box in the workflow diagram.
    
    If tibia_mask is not provided but volume_array is present, automatically
    estimates the tibial plateau landmark to calculate meniscus_extrusion_mm.
    """
    features = {
        "meniscus_volume_cm3": compute_volume_cm3(meniscus_mask, spacing),
        "meniscus_thickness_mm": compute_thickness_mm(meniscus_mask, spacing),
    }

    if tibia_mask is not None:
        features["meniscus_extrusion_mm"] = compute_extrusion_mm(meniscus_mask, tibia_mask, spacing)
    elif volume_array is not None:
        from src.landmarks import detect_tibial_plateau_landmarks
        landmark_res = detect_tibial_plateau_landmarks(volume_array, meniscus_mask, spacing)
        features["meniscus_extrusion_mm"] = landmark_res["meniscus_extrusion_mm"]
    else:
        features["meniscus_extrusion_mm"] = None

    return features
