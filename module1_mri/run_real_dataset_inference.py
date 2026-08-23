"""
run_real_dataset_inference.py
-----------------------------
Runs the trained real-MRI Osteoarthritis Classifier (oa_classifier_real.joblib)
on the real RSNA knee MRI dataset.

Evaluates cohort performance, generates detailed clinical OA assessment reports,
and saves the segmented 3D meniscus masks.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import run_module1
from src.classification import load_classifier, predict_oa, _encode_sex, FEATURE_COLUMNS
from src.io_utils import save_mask
from src.reporting import print_report


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "oa_classifier_real.joblib")
    csv_path = os.path.join(base_dir, "data", "reference_db_real", "labels.csv")

    print("=" * 75)
    print("NEXORA ORTHOAI — RUNNING INFERENCE ON REAL RSNA KNEE MRI DATASET")
    print("=" * 75)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: Dataset labels not found at {csv_path}")
        return

    # Load Model & Dataset
    print(f"1. Loading Pretrained Model: {model_path}")
    clf = load_classifier(model_path)

    print(f"2. Loading Reference Cohort Metadata: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total Real Patient Scans: {len(df)}")
    print(f"   Demographics Summary: Age {df['age'].min()}-{df['age'].max()} yrs | BMI {df['bmi'].min()}-{df['bmi'].max()} | Females: {(df['sex']=='F').sum()}, Males: {(df['sex']=='M').sum()}")

    # Run Batch Predictions Across All Real Scans
    print("\n" + "=" * 75)
    print("3. Cohort Evaluation & Performance Metrics")
    print("=" * 75)

    df_encoded = _encode_sex(df)
    X = df_encoded[FEATURE_COLUMNS]
    y_true = df_encoded["oa_label"]

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, preds)

    print(f"\nOverall Cohort Accuracy : {acc * 100:.1f}%")
    print(f"Overall Cohort ROC-AUC  : {auc:.3f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted Healthy  Predicted OA")
    print(f"  True Healthy         {cm[0,0]:<15d}    {cm[0,1]}")
    print(f"  True OA              {cm[1,0]:<15d}    {cm[1,1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, preds, target_names=["Healthy (No OA)", "Osteoarthritis"], digits=3))

    # Run End-to-End Inference on Individual Clinical Cases
    print("\n" + "=" * 75)
    print("4. Detailed Patient Clinical Reports (Sample Test Scans)")
    print("=" * 75)

    test_indices = [0, 2, 4]  # Representative sample cases
    for idx in test_indices:
        row = df.iloc[idx]
        image_path = os.path.join(base_dir, row["image_path"])

        print(f"\nProcessing Case: {row['subject_id']} (Scan: {os.path.basename(image_path)})")
        print(f"Patient Profile: Age {row['age']} | Sex {row['sex']} | BMI {row['bmi']} | True Label: {'OA' if row['oa_label']==1 else 'Healthy'}")

        if os.path.exists(image_path):
            result = run_module1(
                mri_path=image_path,
                age=row["age"],
                sex=row["sex"],
                bmi=row["bmi"],
                clf=clf,
            )

            print()
            print_report(result["report"])

            # Save segmentation mask for inspection
            mask_out_path = image_path.replace(".nii.gz", "_meniscus_seg.nii.gz")
            save_mask(result["meniscus_mask"], result["preprocessed_image"], mask_out_path)
            print(f"Saved 3D Segmentation Mask: {mask_out_path}")
        else:
            print(f"Warning: Image file not found at {image_path}")

    print("\n" + "=" * 75)
    print("INFERENCE RUN COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
