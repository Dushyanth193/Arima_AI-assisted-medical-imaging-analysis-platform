"""
End-to-end demo of Module 1 using a SYNTHETIC volume (no real MRI file or
SimpleITK needed) - proves segmentation -> feature extraction ->
classification -> reporting all connect correctly. Swap the synthetic
volume for `load_mri()` + real preprocessing once you have actual data.

Run: python demo_end_to_end.py
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.segmentation import segment_meniscus_placeholder
from src.features import extract_features
from src.classification import train_oa_classifier, predict_oa
from src.reporting import build_report, print_report


def make_synthetic_knee_volume(shape=(60, 120, 120), seed=1):
    """Fake a knee-MRI-like volume: background noise + a brighter
    crescent-ish region standing in for the meniscus."""
    rng = np.random.default_rng(seed)
    volume = rng.normal(0, 1, shape).astype(np.float32)

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    crescent = ((yy - center[1])**2 + (xx - center[2])**2 < 25**2) & \
               ((yy - center[1])**2 + (xx - center[2])**2 > 15**2) & \
               (np.abs(zz - center[0]) < 8)
    volume[crescent] += 4.0  # bright region the placeholder segmenter will pick up
    return volume


def make_synthetic_reference_table(n=250, seed=0):
    rng = np.random.default_rng(seed)
    oa_label = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        "meniscus_volume_cm3": rng.normal(4.5, 0.6, n) - oa_label * 0.8,
        "meniscus_thickness_mm": rng.normal(4.0, 0.5, n) - oa_label * 0.5,
        "meniscus_extrusion_mm": rng.normal(1.5, 0.5, n) + oa_label * 2.0,
        "age": rng.normal(55, 10, n) + oa_label * 8,
        "bmi": rng.normal(26, 4, n) + oa_label * 2,
        "sex": rng.choice(["M", "F"], size=n),
        "oa_label": oa_label,
    })


def main():
    print("1) Building synthetic knee volume ...")
    volume = make_synthetic_knee_volume()
    spacing = (1.0, 1.0, 1.0)  # stand-in for real preprocessed spacing

    print("2) Running placeholder segmentation ...")
    mask = segment_meniscus_placeholder(volume)
    print(f"   segmented voxel count: {int(mask.sum())}")

    print("3) Extracting features ...")
    features = extract_features(mask, spacing, tibia_mask=None)
    print(f"   features: {features}")

    print("4) Training OA classifier on synthetic reference DB ...")
    ref_df = make_synthetic_reference_table()
    clf, metrics = train_oa_classifier(ref_df)
    print(f"   validation accuracy={metrics['accuracy']:.3f}, auc={metrics['auc']:.3f}")

    print("5) Predicting OA status for the synthetic 'new patient' ...")
    patient_features = {**features, "age": 62, "bmi": 28, "sex": "F"}
    oa_result = predict_oa(clf, patient_features)

    print("6) Building final report ...")
    report = build_report(patient_id="synthetic_case_001", features=features, oa_result=oa_result)
    print()
    print_report(report)


if __name__ == "__main__":
    main()
