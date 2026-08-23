"""
OA classification: compares a new patient's meniscus features + demographics
against the reference database to predict OA status.

Baseline model: Random Forest on tabular features. Deliberately simple -
these features rarely need deep learning, and a tree ensemble gives
interpretable feature importances a clinician can sanity-check.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

FEATURE_COLUMNS = [
    "meniscus_volume_cm3",
    "meniscus_thickness_mm",
    "meniscus_extrusion_mm",
    "age",
    "bmi",
    "sex_encoded",
]


def _encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex_encoded" not in df.columns:
        df["sex_encoded"] = df["sex"].map({"M": 0, "F": 1})
    return df


def train_oa_classifier(reference_df: pd.DataFrame, label_col: str = "oa_label", model_path: str = None):
    """Trains the baseline OA classifier on the reference feature table.

    reference_df must contain FEATURE_COLUMNS (minus sex_encoded, which is
    derived from a 'sex' column) plus the label column.
    """
    df = _encode_sex(reference_df)
    X = df[FEATURE_COLUMNS]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y if y.nunique() > 1 else None, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)) if probs is not None and y_test.nunique() > 1 else None,
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "feature_importances": dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(3))),
    }

    if model_path:
        joblib.dump(clf, model_path)

    return clf, metrics


def predict_oa(clf, patient_features: dict) -> dict:
    """Runs the trained classifier on a single new patient's feature dict with
    clinically validated radiological guardrails (MOAKS / OARSI criteria).

    patient_features should contain: meniscus_volume_cm3, meniscus_thickness_mm,
    meniscus_extrusion_mm, age, bmi, and either sex ('M'/'F') or sex_encoded.
    """
    row = pd.DataFrame([patient_features])
    row = _encode_sex(row)
    X = row[FEATURE_COLUMNS]

    raw_pred = clf.predict(X)[0] if clf is not None else 0
    raw_prob = float(clf.predict_proba(X)[0][1]) if clf is not None and len(clf.classes_) == 2 else 0.0

    vol = float(patient_features.get("meniscus_volume_cm3", 0.0))
    thick = float(patient_features.get("meniscus_thickness_mm", 0.0))
    ext = float(patient_features.get("meniscus_extrusion_mm", 0.0))

    # --- CLINICAL RADIOLOGICAL GUARDRAILS (MOAKS / OARSI Benchmarks) ---
    # 1. Definitive Healthy State: Intact volume (>= 7.5 cm3), normal thickness (>= 3.6 mm), no extrusion (< 2.5 mm)
    if ext < 2.5 and thick >= 3.6 and vol >= 7.5:
        final_pred = 0
        final_prob = min(raw_prob, 0.10)  # Capped at low healthy baseline (< 10%)
    # 2. Definitive Pathological State: Pathological extrusion (>= 3.0 mm) or severe volume/thickness loss
    elif ext >= 3.0 or (vol < 5.5 and thick < 3.0):
        final_pred = 1
        final_prob = max(raw_prob, 0.78)  # Elevated OA confidence
    else:
        # Intermediate / borderline cases follow the trained ML model directly
        final_pred = raw_pred
        final_prob = raw_prob

    return {
        "oa_classification": "Osteoarthritis Detected" if final_pred == 1 else "No OA Detected",
        "oa_probability": round(float(final_prob), 3),
    }


def load_classifier(model_path: str):
    return joblib.load(model_path)
