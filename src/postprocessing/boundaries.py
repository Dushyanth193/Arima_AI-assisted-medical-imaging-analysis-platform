"""
Boundary Extraction Utilities
Extracts 3D boundary voxels/contours from binary segmentation masks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def extract_surface_boundary(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extracts the outer 1-voxel boundary of a 3D binary mask.
    Boundary = BinaryMask AND NOT ErodedBinaryMask.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)
    boundary = binary_mask.astype(bool) & ~eroded
    return boundary.astype(np.uint8)
