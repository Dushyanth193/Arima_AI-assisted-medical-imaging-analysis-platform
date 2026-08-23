"""
test_new_patients_io.py
-----------------------
Generates test NIfTI (.nii.gz) scans in data/new_patient/, runs them through
the real file I/O, N4 bias field correction, resampling, and normalization
pipeline, and evaluates them with the persisted OA classifier.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.io_utils import load_mri, save_mask
from src.preprocessing import preprocess_pipeline
from src.pipeline import run_module1
from src.classification import load_classifier
from src.reporting import print_report


def generate_new_patient_scan(
    output_path: str,
    meniscus_health: str = "healthy",  # 'healthy', 'mild_oa', 'severe_oa'
    shape: tuple = (36, 72, 72),
    spacing: tuple = (0.75, 0.75, 1.5),
    seed: int = 42,
):
    """Generates a synthetic 3D knee MRI scan (.nii.gz) with realistic
    anisotropic spacing, coil bias field, and anatomical geometry."""
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape, dtype=np.float32)

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    # Knee soft tissue envelope
    soft_tissue = ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 < 30 ** 2) & (np.abs(zz - center[0]) < 16)
    volume[soft_tissue] = 130.0

    # Meniscus geometry based on condition
    if meniscus_health == "healthy":
        outer_r, inner_r, z_thick, intensity = 24, 15, 6, 280.0
    elif meniscus_health == "mild_oa":
        outer_r, inner_r, z_thick, intensity = 20, 15, 4, 250.0
    else:  # severe_oa
        outer_r, inner_r, z_thick, intensity = 18, 15, 3, 230.0

    meniscus = (
        ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 < outer_r ** 2)
        & ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 > inner_r ** 2)
        & (np.abs(zz - center[0]) < z_thick)
    )
    volume[meniscus] = intensity

    # Add realistic spatial bias field gradient
    bias = 1.0 + 0.35 * (yy / shape[1]) + 0.25 * (xx / shape[2])
    bias = np.broadcast_to(bias, shape)
    volume = volume * bias

    # Add Gaussian thermal noise
    noise = rng.normal(0, 4.0, shape).astype(np.float32)
    volume = np.clip(volume + noise, 0, None)

    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(img, output_path)
    return output_path


def main():
    print("=" * 70)
    print("1. Creating Test NIfTI (.nii.gz) Patient Scans in data/new_patient/")
    print("=" * 70)

    patients = [
        {
            "id": "CASE_001_HEALTHY_FEMALE",
            "filename": "case001_healthy_female.nii.gz",
            "condition": "healthy",
            "age": 38,
            "sex": "F",
            "bmi": 21.8,
            "seed": 201,
            "spacing": (0.8, 0.8, 1.5),
        },
        {
            "id": "CASE_002_SEVERE_OA_MALE",
            "filename": "case002_severe_oa_male.nii.gz",
            "condition": "severe_oa",
            "age": 76,
            "sex": "M",
            "bmi": 32.4,
            "seed": 202,
            "spacing": (0.7, 0.7, 2.0),
        },
        {
            "id": "CASE_003_MILD_OA_FEMALE",
            "filename": "case003_mild_oa_female.nii.gz",
            "condition": "mild_oa",
            "age": 63,
            "sex": "F",
            "bmi": 28.1,
            "seed": 203,
            "spacing": (0.75, 0.75, 1.2),
        },
    ]

    out_dir = "data/new_patient"
    for p in patients:
        filepath = os.path.join(out_dir, p["filename"]).replace("\\", "/")
        p["filepath"] = filepath
        generate_new_patient_scan(
            filepath,
            meniscus_health=p["condition"],
            spacing=p["spacing"],
            seed=p["seed"],
        )
        print(f"Created: {filepath} ({p['condition']}) with raw spacing {p['spacing']}")

    print("\n" + "=" * 70)
    print("2. Validating SimpleITK I/O, N4 Bias Correction & Resampling")
    print("=" * 70)
    for p in patients:
        raw = load_mri(p["filepath"])
        print(f"\nPatient: {p['id']}")
        print(f"  Raw NIfTI Array Shape: {raw['array'].shape}, Spacing: {raw['spacing']}")

        preprocessed_img, norm_array = preprocess_pipeline(
            raw["sitk_image"],
            target_spacing=(0.5, 0.5, 0.5),
            bias_correct=True,
            denoise_flag=True,
        )
        print(f"  Preprocessed Image Size: {preprocessed_img.GetSize()} @ {preprocessed_img.GetSpacing()} mm")
        print(f"  Normalized Array Shape : {norm_array.shape} (Mean: {norm_array.mean():.4f}, Std: {norm_array.std():.4f})")

    print("\n" + "=" * 70)
    print("3. Running End-to-End Module 1 Inference with Persisted Classifier")
    print("=" * 70)
    clf = load_classifier("models/oa_classifier.joblib")

    for p in patients:
        result = run_module1(
            mri_path=p["filepath"],
            age=p["age"],
            sex=p["sex"],
            bmi=p["bmi"],
            clf=clf,
        )

        # Save segmented meniscus mask alongside the patient scan
        mask_out_path = p["filepath"].replace(".nii.gz", "_meniscus_seg.nii.gz")
        save_mask(result["meniscus_mask"], result["preprocessed_image"], mask_out_path)

        print("\n" + "-" * 50)
        print(f"Patient ID: {p['id']} (Age: {p['age']}, Sex: {p['sex']}, BMI: {p['bmi']})")
        print_report(result["report"])
        print(f"Saved Segmentation Mask: {mask_out_path}")


if __name__ == "__main__":
    main()
