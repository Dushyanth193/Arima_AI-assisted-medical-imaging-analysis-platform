"""
Preprocessing pipeline shared by BOTH the reference-database build and the
new-patient inference path. Keep it one shared function - if these two
paths ever drift apart, every downstream comparison becomes invalid.
"""

import SimpleITK as sitk
import numpy as np


def resample_to_spacing(image: sitk.Image, target_spacing=(0.5, 0.5, 0.5), is_mask=False) -> sitk.Image:
    """Resample to isotropic spacing so voxel-based measurements are comparable
    across scanners/protocols."""
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
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)

    return resampler.Execute(image)


def denoise(image: sitk.Image, iterations=5, conductance=3.0) -> sitk.Image:
    """Edge-preserving denoise. Cheap and effective first pass; swap for
    N4 + non-local-means later if the segmentation model needs it."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.CurvatureAnisotropicDiffusion(
        image_f, timeStep=0.0625, numberOfIterations=iterations, conductanceParameter=conductance
    )


def n4_bias_correction(image: sitk.Image, mask: sitk.Image = None) -> sitk.Image:
    """Corrects MRI intensity inhomogeneity (bias field). Optional - turn on
    if you see systematic intensity drift across the volume."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(image_f, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(image_f, mask)


def zscore_normalize(array: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Normalize intensities to zero mean / unit variance, optionally
    computing stats only within a foreground mask (recommended once you
    have a rough knee mask - avoids background voxels skewing the stats)."""
    vals = array[mask > 0] if mask is not None else array
    mean, std = float(vals.mean()), float(vals.std())
    std = std if std > 1e-8 else 1.0
    return (array - mean) / std


def preprocess_pipeline(
    image: sitk.Image,
    target_spacing=(0.5, 0.5, 0.5),
    denoise_flag=True,
    bias_correct=False,
):
    """Single entry point used by both the reference-DB builder and the
    new-patient path. Returns (preprocessed sitk.Image, normalized numpy array)."""
    img = resample_to_spacing(image, target_spacing, is_mask=False)

    if bias_correct:
        img = n4_bias_correction(img)

    if denoise_flag:
        img = denoise(img)

    array = sitk.GetArrayFromImage(img)
    norm_array = zscore_normalize(array)

    return img, norm_array
