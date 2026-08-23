"""
Medial meniscus segmentation.

Two paths are provided:

1. segment_with_nnunet()          -> the real production path. Requires
   `pip install nnunetv2`, a trained model in nnUNet_results, and GPU.
   This is what the diagram's "Medial Meniscus Detection and Segmentation"
   box maps to.

2. segment_meniscus_placeholder() -> a lightweight intensity/morphology
   based stand-in. NOT clinically meaningful - it exists purely so you can
   build and test the rest of the pipeline (features, classifier, reporting)
   before nnU-Net is trained. Swap it out for (1) once you have weights.
"""

import os
import subprocess
import numpy as np
from skimage import morphology, measure


def segment_with_nnunet(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    config: str = "3d_fullres",
    fold: str = "all",
) -> str:
    """Runs nnU-Net v2 inference via its CLI.

    input_folder must contain files named like `<case_id>_0000.nii.gz`
    (nnU-Net's required naming convention for a single input channel).
    Requires nnUNet_raw / nnUNet_preprocessed / nnUNet_results env vars
    to be set, and a model already trained for `dataset_id`.
    """
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", str(dataset_id),
        "-c", config,
        "-f", fold,
    ]
    subprocess.run(cmd, check=True)
    return output_folder


def segment_meniscus_placeholder(
    volume_array: np.ndarray,
    intensity_percentile: float = 90,
    min_size: int = 50,
) -> np.ndarray:
    """Intensity threshold + morphological cleanup + largest-component
    selection. Good enough to produce a plausible-shaped mask for pipeline
    testing; not a substitute for a trained segmentation model.
    """
    threshold = np.percentile(volume_array, intensity_percentile)
    mask = volume_array >= threshold

    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, morphology.ball(1))

    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    largest_label = int(np.argmax(counts))
    mask = labeled == largest_label

    return mask.astype(np.uint8)
