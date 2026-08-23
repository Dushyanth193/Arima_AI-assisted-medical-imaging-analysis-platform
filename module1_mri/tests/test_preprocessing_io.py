"""
tests/test_preprocessing_io.py
------------------------------
Unit and integration tests for MRI file I/O (SimpleITK NIfTI reading/writing),
resampling, N4 bias field correction, edge-preserving denoising, and
z-score intensity normalization in src/preprocessing.py and src/io_utils.py.
"""

import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io_utils import load_mri, save_mask
from src.preprocessing import (
    resample_to_spacing,
    denoise,
    n4_bias_correction,
    zscore_normalize,
    preprocess_pipeline,
)


def create_synthetic_knee_phantom(
    shape=(30, 60, 60),
    spacing=(0.8, 0.8, 1.5),
    add_bias=False,
    noise_std=0.0,
    seed=42,
) -> sitk.Image:
    """Creates a synthetic 3D knee MRI scan with tissue contrast, geometry,
    optional spatial bias field, and noise."""
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape, dtype=np.float32)

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    # Knee tissue layers: Bone marrow / soft tissue background
    soft_tissue = ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 < 24 ** 2) & (np.abs(zz - center[0]) < 12)
    volume[soft_tissue] = 120.0

    # Meniscus crescent (higher signal intensity)
    meniscus = (
        ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 < 20 ** 2)
        & ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 > 14 ** 2)
        & (np.abs(zz - center[0]) < 4)
    )
    volume[meniscus] = 260.0

    # Optional low-frequency spatial intensity bias (e.g. coil sensitivity gradient)
    if add_bias:
        bias_field = 1.0 + 0.5 * (yy / shape[1]) + 0.3 * (xx / shape[2])
        volume = volume * bias_field

    # Optional Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0, noise_std, shape).astype(np.float32)
        volume = np.clip(volume + noise, 0, None)

    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    img.SetOrigin((10.0, 20.0, 30.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def test_load_and_save_mri_io():
    """Validates SimpleITK reading of .nii.gz and mask writing with geometry preservation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_scan.nii.gz")
        mask_out_path = os.path.join(tmpdir, "test_mask.nii.gz")

        orig_img = create_synthetic_knee_phantom(shape=(25, 50, 50), spacing=(0.9, 0.9, 1.6))
        sitk.WriteImage(orig_img, test_path)

        # 1. Load scan via load_mri
        loaded = load_mri(test_path)
        assert "array" in loaded
        assert "spacing" in loaded
        assert "origin" in loaded
        assert "sitk_image" in loaded

        assert loaded["array"].shape == (25, 50, 50)
        assert np.allclose(loaded["spacing"], (0.9, 0.9, 1.6))
        assert np.allclose(loaded["origin"], (10.0, 20.0, 30.0))

        # 2. Save mask via save_mask and verify metadata
        dummy_mask = (loaded["array"] > 150).astype(np.uint8)
        save_mask(dummy_mask, loaded["sitk_image"], mask_out_path)

        saved_mask_img = sitk.ReadImage(mask_out_path)
        assert np.allclose(saved_mask_img.GetSpacing(), (0.9, 0.9, 1.6))
        assert np.allclose(saved_mask_img.GetOrigin(), (10.0, 20.0, 30.0))
        assert sitk.GetArrayFromImage(saved_mask_img).sum() == dummy_mask.sum()
        print(f"  MRI I/O validated: array shape={loaded['array'].shape}, spacing={loaded['spacing']}")


def test_resample_to_spacing():
    """Validates resampling from anisotropic spacing to isotropic target spacing."""
    orig_shape = (20, 40, 40)
    orig_spacing = (1.0, 1.0, 2.0)  # FOV: 40mm x 40mm x 40mm
    target_spacing = (0.5, 0.5, 0.5)

    img = create_synthetic_knee_phantom(shape=orig_shape, spacing=orig_spacing)
    resampled = resample_to_spacing(img, target_spacing=target_spacing)

    new_size = resampled.GetSize()  # (x, y, z)
    new_spacing = resampled.GetSpacing()

    print(f"  Original size: {img.GetSize()} @ spacing {orig_spacing}")
    print(f"  Resampled size: {new_size} @ spacing {new_spacing}")

    assert np.allclose(new_spacing, target_spacing)
    # Expected size: (40 * 1.0 / 0.5, 40 * 1.0 / 0.5, 20 * 2.0 / 0.5) = (80, 80, 80)
    assert new_size == (80, 80, 80)


def test_n4_bias_field_correction():
    """Validates that N4 bias correction flattens simulated spatial intensity inhomogeneity."""
    shape = (24, 48, 48)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (((yy - 24) ** 2 + (xx - 24) ** 2 < 18 ** 2) & (np.abs(zz - 12) < 8))
    bias = 1.0 + 0.4 * (yy / 48.0) + 0.3 * (xx / 48.0)
    bias = np.broadcast_to(bias, shape)

    vol = np.zeros(shape, dtype=np.float32)
    vol[mask] = 100.0 * bias[mask]

    img_biased = sitk.GetImageFromArray(vol)
    img_biased.SetSpacing((1.0, 1.0, 1.0))

    std_before = float(vol[mask].std())
    img_corrected = n4_bias_correction(img_biased)
    arr_after = sitk.GetArrayFromImage(img_corrected)
    std_after = float(arr_after[mask].std())

    print(f"  Tissue intensity std before N4: {std_before:.2f}, after N4: {std_after:.2f}")
    assert std_after < std_before / 2.0, "N4 bias correction should dramatically reduce spatial intensity variance across homogeneous tissue"


def test_denoise_reduces_noise():
    """Validates that edge-preserving curvature anisotropic diffusion reduces noise variance."""
    img_noisy = create_synthetic_knee_phantom(shape=(20, 40, 40), spacing=(1.0, 1.0, 1.0), noise_std=15.0)
    arr_noisy = sitk.GetArrayFromImage(img_noisy)

    img_denoised = denoise(img_noisy, iterations=5, conductance=3.0)
    arr_denoised = sitk.GetArrayFromImage(img_denoised)

    # Test noise reduction in homogeneous background / soft-tissue region
    sample_roi_noisy = arr_noisy[8:12, 18:22, 18:22]
    sample_roi_denoised = arr_denoised[8:12, 18:22, 18:22]

    var_noisy = float(sample_roi_noisy.var())
    var_denoised = float(sample_roi_denoised.var())

    print(f"  ROI variance before denoise: {var_noisy:.2f}, after denoise: {var_denoised:.2f}")
    assert var_denoised < var_noisy, "Denoising must reduce high-frequency local variance"


def test_zscore_normalization():
    """Validates zero-mean, unit-variance normalization."""
    arr = np.random.normal(loc=125.0, scale=45.0, size=(20, 30, 30)).astype(np.float32)

    # 1. Global normalization
    norm_global = zscore_normalize(arr)
    print(f"  Global normalized mean: {norm_global.mean():.4f}, std: {norm_global.std():.4f}")
    assert abs(norm_global.mean()) < 1e-4
    assert abs(norm_global.std() - 1.0) < 1e-3

    # 2. Masked normalization
    mask = (arr > 120.0).astype(np.uint8)
    norm_masked = zscore_normalize(arr, mask=mask)
    masked_vals = norm_masked[mask > 0]
    print(f"  Masked normalized mean: {masked_vals.mean():.4f}, std: {masked_vals.std():.4f}")
    assert abs(masked_vals.mean()) < 1e-4
    assert abs(masked_vals.std() - 1.0) < 1e-3


def test_full_preprocess_pipeline():
    """Validates complete preprocessing pipeline with all flags enabled."""
    img = create_synthetic_knee_phantom(
        shape=(20, 40, 40),
        spacing=(0.8, 0.8, 1.2),
        add_bias=True,
        noise_std=5.0,
    )

    preprocessed_img, norm_array = preprocess_pipeline(
        img,
        target_spacing=(0.5, 0.5, 0.5),
        denoise_flag=True,
        bias_correct=True,
    )

    assert isinstance(preprocessed_img, sitk.Image)
    assert isinstance(norm_array, np.ndarray)
    assert np.allclose(preprocessed_img.GetSpacing(), (0.5, 0.5, 0.5))
    assert abs(norm_array.mean()) < 1e-2
    assert abs(norm_array.std() - 1.0) < 1e-2
    print(f"  Full pipeline output image size: {preprocessed_img.GetSize()}, array shape: {norm_array.shape}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  PASSED\n")
    print(f"All {len(tests)} I/O & preprocessing tests passed.")
