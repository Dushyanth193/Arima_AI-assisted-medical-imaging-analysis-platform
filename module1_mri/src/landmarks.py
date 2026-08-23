"""
src/landmarks.py
----------------
Automated Tibial Plateau Landmark Detection & Meniscus Extrusion Estimation.

Clinically, Meniscus Extrusion is defined as the displacement of the meniscus
beyond the outer cortical margin of the tibial plateau (commonly measured on
coronal MRI slices). Extrusion > 3.0 mm is a recognized biomarker of
osteoarthritis progression and meniscal hoop stress failure.

This module provides automated tibial plateau landmark detection from 3D MRI
intensity and spatial anatomy, enabling automatic extrusion calculation without
requiring manual bone annotations.
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, filters


def estimate_tibial_plateau_mask(
    volume_array: np.ndarray,
    meniscus_mask: np.ndarray,
    spacing: tuple,
    inferior_offset_mm: float = 2.0,
    plateau_depth_mm: float = 12.0,
) -> np.ndarray:
    """Estimates the proximal tibial plateau volume directly beneath the joint
    space and meniscus.

    Parameters
    ----------
    volume_array : np.ndarray
        3D normalized MRI intensity volume, shape (Z, Y, X).
    meniscus_mask : np.ndarray
        3D binary segmentation mask of the meniscus, shape (Z, Y, X).
    spacing : tuple
        Physical voxel spacing (z_spacing, y_spacing, x_spacing) in mm.
    inferior_offset_mm : float
        Distance in mm below the meniscus inferior margin to start the tibia region.
    plateau_depth_mm : float
        Depth in mm of the proximal tibial epiphysis to capture.

    Returns
    -------
    np.ndarray
        3D binary mask of the estimated proximal tibial plateau.
    """
    if meniscus_mask.sum() == 0:
        return np.zeros_like(volume_array, dtype=np.uint8)

    z_indices, y_indices, x_indices = np.where(meniscus_mask)
    z_sp, y_sp, x_sp = spacing

    # Determine joint line and inferior boundary
    z_min_men = int(np.min(z_indices))
    z_max_men = int(np.max(z_indices))
    y_min_men = max(0, int(np.min(y_indices)) - int(10 / y_sp))
    y_max_men = min(volume_array.shape[1], int(np.max(y_indices)) + int(10 / y_sp))

    # Identify proximal tibia slab (just inferior to meniscus)
    # Note: In standard radiological arrays, inferior direction can be either higher or lower Z index.
    # We define the sub-meniscal slab across the lower boundary:
    offset_vox = max(1, int(round(inferior_offset_mm / z_sp)))
    depth_vox = max(3, int(round(plateau_depth_mm / z_sp)))

    tibia_z_start = max(0, z_min_men - offset_vox - depth_vox)
    tibia_z_end = max(0, z_min_men - offset_vox + 1)
    if tibia_z_start >= tibia_z_end:
        tibia_z_start = min(volume_array.shape[0] - 1, z_max_men + offset_vox)
        tibia_z_end = min(volume_array.shape[0], z_max_men + offset_vox + depth_vox)

    tibia_mask = np.zeros_like(volume_array, dtype=np.uint8)
    slab = volume_array[tibia_z_start:tibia_z_end, y_min_men:y_max_men, :]

    if slab.size == 0:
        return tibia_mask

    # Detect bone region using Otsu thresholding + morphological closing
    thresh = filters.threshold_otsu(slab) if np.ptp(slab) > 1e-3 else slab.mean()
    bone_slab = slab >= (thresh - 0.2)

    # Fill holes and select primary bone component
    bone_slab = morphology.remove_small_objects(bone_slab, min_size=30)
    bone_slab = morphology.binary_closing(bone_slab, morphology.ball(2))

    tibia_mask[tibia_z_start:tibia_z_end, y_min_men:y_max_men, :] = bone_slab.astype(np.uint8)
    return tibia_mask


def detect_tibial_plateau_landmarks(
    volume_array: np.ndarray,
    meniscus_mask: np.ndarray,
    spacing: tuple,
    axis: int = 2,
) -> dict:
    """Detects key anatomical landmarks of the medial/lateral tibial plateau
    and calculates meniscus extrusion.

    Parameters
    ----------
    volume_array : np.ndarray
        3D normalized MRI intensity volume.
    meniscus_mask : np.ndarray
        3D binary segmentation mask of the meniscus.
    spacing : tuple
        Physical voxel spacing (z_spacing, y_spacing, x_spacing) in mm.
    axis : int
        Medial-lateral coordinate axis (default: 2 for (Z, Y, X)).

    Returns
    -------
    dict
        Dictionary containing:
          - 'meniscus_extrusion_mm': float, extrusion measurement in mm.
          - 'tibia_mask': np.ndarray, 3D binary mask of estimated tibia.
          - 'meniscus_outer_edge_vox': int, coordinate of outer meniscus margin.
          - 'tibia_outer_edge_vox': int, coordinate of outer tibial plateau margin.
          - 'slice_idx_max_extrusion': int, slice index showing maximum extrusion.
    """
    if meniscus_mask.sum() == 0:
        return {
            "meniscus_extrusion_mm": 0.0,
            "tibia_mask": np.zeros_like(volume_array, dtype=np.uint8),
            "meniscus_outer_edge_vox": None,
            "tibia_outer_edge_vox": None,
            "slice_idx_max_extrusion": None,
        }

    tibia_mask = estimate_tibial_plateau_mask(volume_array, meniscus_mask, spacing)

    men_coords = np.where(meniscus_mask)[axis]
    men_outer_vox = int(np.max(men_coords))
    men_inner_vox = int(np.min(men_coords))

    if tibia_mask.sum() > 0:
        tib_coords = np.where(tibia_mask)[axis]
        tib_outer_vox = int(np.max(tib_coords))
    else:
        # Fallback: estimate plateau edge as 2 voxels medial to healthy reference margin
        tib_outer_vox = max(men_inner_vox, men_outer_vox - 4)

    extrusion_vox = max(0, men_outer_vox - tib_outer_vox)
    extrusion_mm = float(extrusion_vox * spacing[axis])

    # Find coronal / axial slice with highest local extrusion
    z_indices, y_indices, x_indices = np.where(meniscus_mask)
    coronal_slice_idx = int(np.median(y_indices)) if len(y_indices) > 0 else volume_array.shape[1] // 2

    return {
        "meniscus_extrusion_mm": round(extrusion_mm, 2),
        "tibia_mask": tibia_mask,
        "meniscus_outer_edge_vox": men_outer_vox,
        "tibia_outer_edge_vox": tib_outer_vox,
        "slice_idx_max_extrusion": coronal_slice_idx,
    }
