"""
Mask Postprocessing Utilities
Clean segmentation masks using connected-component analysis, morphological closing,
and hole filling.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import morphology


def clean_binary_mask(
    mask: np.ndarray,
    min_component_size: int = 500,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Cleans a binary 3D segmentation mask:
    1. Keeps only components larger than min_component_size (or the largest component if none match).
    2. Performs binary hole filling per axial slice or full 3D volume.
    """
    if not np.any(mask):
        return mask

    # Connected component labeling
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask

    # Calculate component sizes
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    clean_mask = np.zeros_like(mask, dtype=bool)

    # Keep components > min_component_size, or at least the largest one
    largest_label = np.argmax(sizes) + 1
    for label_idx, size in enumerate(sizes, start=1):
        if size >= min_component_size or label_idx == largest_label:
            clean_mask |= (labeled == label_idx)

    # Optional 3D hole filling
    if fill_holes:
        clean_mask = ndimage.binary_fill_holes(clean_mask)

    return clean_mask.astype(np.uint8)


def clean_multiclass_mask(
    mask: np.ndarray,
    num_classes: int = 3,
    min_component_size: int = 500,
) -> np.ndarray:
    """
    Cleans each non-background label in a multi-class mask independently
    and recombines them.
    """
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for class_id in range(1, num_classes):
        binary = (mask == class_id)
        if np.any(binary):
            cleaned_binary = clean_binary_mask(
                binary, min_component_size=min_component_size, fill_holes=True
            )
            cleaned[cleaned_binary > 0] = class_id
    return cleaned
