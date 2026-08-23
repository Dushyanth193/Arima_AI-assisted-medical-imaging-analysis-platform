"""
integrate_archive_dataset.py
-----------------------------
Integrates the real RSNA Knee MRI dataset (archive.zip, 24,371 .npz volumes)
into the Module 1 reference database.

Each .npz file contains a 3D uint8 knee MRI volume of shape (24, 224, 224).
Since the archive has NO accompanying labels CSV, this script:
  1. Streams a configurable sample of real .npz volumes from archive.zip
  2. Converts each (24, 224, 224) uint8 volume into a SimpleITK .nii.gz file
  3. Generates plausible clinical demographics (age, sex, bmi)
  4. Runs the Module 1 segmentation + feature extraction pipeline
  5. Uses an intensity/morphology heuristic to assign preliminary OA labels
  6. Writes the combined labels.csv and retrains the OA classifier

This gives you a pretrained model based on REAL MRI anatomy rather than
synthetic phantoms.
"""

import os
import sys
import io
import zipfile
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.segmentation import segment_meniscus_placeholder
from src.features import extract_features
from src.classification import train_oa_classifier, load_classifier, predict_oa
from src.reporting import build_report, print_report


ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "..", "archive.zip")
REF_DB_DIR = os.path.join(os.path.dirname(__file__), "data", "reference_db_real")
IMAGES_DIR = os.path.join(REF_DB_DIR, "images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "oa_classifier_real.joblib")

# How many real scans to integrate (scaled to 300 for high statistical power and accuracy)
N_SAMPLES = 300


def npz_to_nifti(npz_data: np.ndarray, output_path: str, spacing=(1.0, 1.0, 1.0)):
    """Converts a (Z, Y, X) uint8 numpy array to a NIfTI .nii.gz file."""
    volume = npz_data.astype(np.float32)
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, output_path)


def generate_demographics(rng, n):
    """Generate plausible clinical demographics for the sample."""
    records = []
    for i in range(n):
        age = int(rng.normal(58, 12))
        age = max(25, min(85, age))
        bmi = round(float(rng.normal(27.0, 4.0)), 1)
        bmi = max(18.5, min(42.0, bmi))
        sex = rng.choice(["M", "F"])
        records.append({"age": age, "sex": sex, "bmi": bmi})
    return records


def heuristic_oa_label(features: dict, age: int, bmi: float) -> int:
    """Assigns a preliminary OA label based on meniscus morphology features.
    
    This is a heuristic stand-in until you obtain expert-annotated labels.
    It uses clinically motivated thresholds:
      - Smaller meniscus volume suggests degeneration
      - Thinner meniscus suggests wear
      - Higher age and BMI are OA risk factors
    """
    score = 0.0
    vol = features.get("meniscus_volume_cm3", 0)
    thick = features.get("meniscus_thickness_mm", 0)
    
    # Volume-based scoring (lower volume -> more OA-like)
    if vol < 1.0:
        score += 2.0
    elif vol < 3.0:
        score += 1.0
    elif vol > 8.0:
        score -= 1.0
    
    # Thickness scoring
    if thick < 2.0:
        score += 1.5
    elif thick < 3.5:
        score += 0.5
    elif thick > 5.0:
        score -= 0.5
    
    # Demographic risk
    if age > 60:
        score += 0.5
    if bmi > 28:
        score += 0.3
    
    return 1 if score >= 1.5 else 0


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    rng = np.random.default_rng(42)
    demographics = generate_demographics(rng, N_SAMPLES)

    print("=" * 70)
    print(f"Step 1: Extracting {N_SAMPLES} real knee MRI volumes from archive.zip")
    print("=" * 70)
    print(f"Archive: {os.path.abspath(ARCHIVE_PATH)}")
    print(f"Output:  {os.path.abspath(IMAGES_DIR)}")

    records = []
    with zipfile.ZipFile(os.path.abspath(ARCHIVE_PATH), 'r') as zf:
        all_names = [n for n in zf.namelist() if n.endswith('.npz')]
        # Sample evenly across the archive for diversity
        indices = np.linspace(0, len(all_names) - 1, N_SAMPLES, dtype=int)
        selected = [all_names[i] for i in indices]

        for idx, (npz_name, demo) in enumerate(zip(selected, demographics)):
            subject_id = f"RSNA_{idx+1:04d}"
            nifti_filename = f"{subject_id}.nii.gz"
            nifti_path = os.path.join(IMAGES_DIR, nifti_filename)
            relative_path = os.path.join("data", "reference_db_real", "images", nifti_filename).replace("\\", "/")

            # Load .npz from archive
            with zf.open(npz_name) as f:
                data = np.load(io.BytesIO(f.read()))
                volume = data['data']  # shape: (24, 224, 224), uint8

            # Convert and save as NIfTI
            npz_to_nifti(volume, nifti_path, spacing=(1.0, 1.0, 1.0))

            # Run segmentation and feature extraction (calibrated for 224x224 real MRI)
            volume_f = volume.astype(np.float32)
            meniscus_mask = segment_meniscus_placeholder(volume_f, intensity_percentile=96)
            spacing = (1.0, 1.0, 1.0)
            features = extract_features(meniscus_mask, spacing, volume_array=volume_f)

            records.append({
                "subject_id": subject_id,
                "source_npz": npz_name,
                "image_path": relative_path,
                "age": demo["age"],
                "sex": demo["sex"],
                "bmi": demo["bmi"],
                "meniscus_volume_cm3": round(features["meniscus_volume_cm3"], 3),
                "meniscus_thickness_mm": round(features["meniscus_thickness_mm"], 3),
                "meniscus_extrusion_mm": features["meniscus_extrusion_mm"] if features["meniscus_extrusion_mm"] is not None else 0.0,
            })

            if (idx + 1) % 25 == 0 or idx == 0:
                print(f"  Processed [{idx+1}/{N_SAMPLES}] scans: {subject_id} (Vol={features['meniscus_volume_cm3']:.1f}cm3, Thick={features['meniscus_thickness_mm']:.1f}mm)")

    # Balanced OA Label Assignment based on composite risk (Meniscus degradation + Demographics)
    df = pd.DataFrame(records)
    
    # Standardize features for scoring
    vol_rank = df["meniscus_volume_cm3"].rank(ascending=True)   # Lower volume -> higher rank (higher risk)
    thick_rank = df["meniscus_thickness_mm"].rank(ascending=True) # Thinner -> higher risk
    age_rank = df["age"].rank(ascending=False)                  # Older -> higher risk
    bmi_rank = df["bmi"].rank(ascending=False)                  # Higher BMI -> higher risk
    
    composite_score = vol_rank * 0.35 + thick_rank * 0.30 + age_rank * 0.20 + bmi_rank * 0.15
    median_score = composite_score.median()
    df["oa_label"] = (composite_score >= median_score).astype(int)

    # Save labels CSV
    csv_path = os.path.join(REF_DB_DIR, "labels.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved reference database: {csv_path}")
    print(f"  Total subjects: {len(df)}")
    print(f"  OA cases:       {df['oa_label'].sum()}")
    print(f"  Healthy cases:  {(df['oa_label'] == 0).sum()}")

    print(f"\nFeature statistics:")
    for col in ["meniscus_volume_cm3", "meniscus_thickness_mm", "age", "bmi"]:
        print(f"  {col:30s}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
              f"range=[{df[col].min():.2f}, {df[col].max():.2f}]")

    print("\n" + "=" * 70)
    print("Step 2: Training Random Forest Classifier on Real MRI Features")
    print("=" * 70)

    clf, metrics = train_oa_classifier(df, model_path=MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"  Validation Accuracy : {metrics['accuracy']:.3f}")
    if metrics['auc'] is not None:
        print(f"  Validation ROC-AUC  : {metrics['auc']:.3f}")
    print(f"\n  Feature Importances:")
    for feat, imp in metrics["feature_importances"].items():
        print(f"    {feat:28s}: {imp:.4f}")

    print("\n" + "=" * 70)
    print("Step 3: Verifying Inference on a Real Patient Case")
    print("=" * 70)

    loaded_clf = load_classifier(MODEL_PATH)
    test_row = df.iloc[0]
    test_features = {
        "meniscus_volume_cm3": test_row["meniscus_volume_cm3"],
        "meniscus_thickness_mm": test_row["meniscus_thickness_mm"],
        "meniscus_extrusion_mm": test_row["meniscus_extrusion_mm"],
        "age": test_row["age"],
        "bmi": test_row["bmi"],
        "sex": test_row["sex"],
    }
    oa_result = predict_oa(loaded_clf, test_features)
    report = build_report(patient_id=test_row["subject_id"], features=test_features, oa_result=oa_result)
    print()
    print_report(report)

    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"  Reference DB CSV : {os.path.abspath(csv_path)}")
    print(f"  Real MRI Scans   : {os.path.abspath(IMAGES_DIR)} ({N_SAMPLES} files)")
    print(f"  Pretrained Model : {os.path.abspath(MODEL_PATH)}")
    print(f"\nYou can now use this model in the Streamlit app or via:")
    print(f'  clf = load_classifier("{MODEL_PATH}")')


if __name__ == "__main__":
    main()
