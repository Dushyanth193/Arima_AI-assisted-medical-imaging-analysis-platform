"""
Orchestrates Module 1 end to end: load -> preprocess -> segment -> extract
features -> classify -> report. Kept deliberately thin - all real logic
lives in the individual modules so each stage can be tested in isolation.

This mirrors the two branches in the workflow diagram:
  - build_reference_database(): runs the pipeline over the reference cohort
    once, to produce the feature table the classifier trains on.
  - run_module1(): runs the same pipeline on a single new patient at
    inference time.
"""

import pandas as pd

from src.io_utils import load_mri, load_reference_database
from src.preprocessing import preprocess_pipeline
from src.segmentation import segment_meniscus_placeholder
from src.features import extract_features
from src.classification import predict_oa
from src.reporting import build_report


def _process_single_case(image_path: str, tibia_mask=None):
    """Shared by both the reference-DB build and new-patient inference -
    this is the one preprocessing/segmentation/feature path both use."""
    raw = load_mri(image_path)
    preprocessed_img, norm_array = preprocess_pipeline(raw["sitk_image"])

    meniscus_mask = segment_meniscus_placeholder(norm_array)

    # sitk spacing is (x, y, z); numpy array is (z, y, x) - reverse to match
    spacing_zyx = preprocessed_img.GetSpacing()[::-1]

    features = extract_features(
        meniscus_mask,
        spacing_zyx,
        tibia_mask=tibia_mask,
        volume_array=norm_array,
    )
    
    from src.landmarks import estimate_tibial_plateau_mask
    estimated_tibia = tibia_mask if tibia_mask is not None else estimate_tibial_plateau_mask(norm_array, meniscus_mask, spacing_zyx)

    return features, meniscus_mask, norm_array, preprocessed_img, estimated_tibia


def build_reference_database(csv_path: str) -> pd.DataFrame:
    """Runs the pipeline over every case in the reference DB CSV and returns
    a feature table ready for train_oa_classifier()."""
    meta_df = load_reference_database(csv_path)

    rows = []
    for _, row in meta_df.iterrows():
        features, meniscus_mask, norm_array, preprocessed_img, estimated_tibia = _process_single_case(row["image_path"])
        rows.append({
            "subject_id": row["subject_id"],
            "age": row["age"],
            "sex": row["sex"],
            "bmi": row["bmi"],
            "oa_label": row["oa_label"],
            **features,
        })

    return pd.DataFrame(rows)


def run_module1(mri_path: str, age: float, sex: str, bmi: float, clf, tibia_mask=None) -> dict:
    """Single public entry point for a new patient - this is the function
    an integration layer (or the future Integrated AI Report) should call."""
    features, meniscus_mask, norm_array, preprocessed_img, estimated_tibia = _process_single_case(mri_path, tibia_mask)

    features_for_model = {
        **features,
        "age": age,
        "sex": sex,
        "bmi": bmi,
    }

    oa_result = predict_oa(clf, features_for_model)
    report = build_report(patient_id=mri_path, features=features, oa_result=oa_result)

    return {
        "report": report,
        "features": features,
        "oa_result": oa_result,
        "meniscus_mask": meniscus_mask,
        "tibia_mask": estimated_tibia,
        "preprocessed_array": norm_array,
        "preprocessed_image": preprocessed_img,
    }
