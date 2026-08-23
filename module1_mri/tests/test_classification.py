"""
Validates the classifier trains and predicts without errors, using a
synthetic reference table where OA cases are constructed to have lower
meniscus volume / higher extrusion (a real, published pattern) so a
working classifier should score meaningfully above chance.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.classification import train_oa_classifier, predict_oa


def make_synthetic_reference_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    oa_label = rng.integers(0, 2, size=n)

    # OA cases: lower volume, higher extrusion, thinner meniscus, older age
    volume = rng.normal(4.5, 0.6, n) - oa_label * 0.8
    thickness = rng.normal(4.0, 0.5, n) - oa_label * 0.5
    extrusion = rng.normal(1.5, 0.5, n) + oa_label * 2.0
    age = rng.normal(55, 10, n) + oa_label * 8
    bmi = rng.normal(26, 4, n) + oa_label * 2
    sex = rng.choice(["M", "F"], size=n)

    df = pd.DataFrame({
        "meniscus_volume_cm3": volume,
        "meniscus_thickness_mm": thickness,
        "meniscus_extrusion_mm": extrusion,
        "age": age,
        "bmi": bmi,
        "sex": sex,
        "oa_label": oa_label,
    })
    return df


def test_classifier_trains_and_beats_chance():
    df = make_synthetic_reference_table(n=300)
    clf, metrics = train_oa_classifier(df)
    print(f"  accuracy={metrics['accuracy']:.3f}, auc={metrics['auc']:.3f}")
    print(f"  feature_importances={metrics['feature_importances']}")
    assert metrics["accuracy"] > 0.6, "Classifier should beat random guessing on separable synthetic data"
    assert metrics["auc"] > 0.6


def test_predict_oa_on_single_patient():
    df = make_synthetic_reference_table(n=300)
    clf, _ = train_oa_classifier(df)

    # construct a patient that looks strongly OA-like
    oa_like_patient = {
        "meniscus_volume_cm3": 3.2,
        "meniscus_thickness_mm": 3.0,
        "meniscus_extrusion_mm": 3.8,
        "age": 68,
        "bmi": 30,
        "sex": "F",
    }
    result = predict_oa(clf, oa_like_patient)
    print(f"  OA-like patient -> {result}")
    assert result["oa_classification"] in ("Osteoarthritis Detected", "No OA Detected")
    assert 0.0 <= result["oa_probability"] <= 1.0

    healthy_like_patient = {
        "meniscus_volume_cm3": 5.2,
        "meniscus_thickness_mm": 4.5,
        "meniscus_extrusion_mm": 0.8,
        "age": 35,
        "bmi": 23,
        "sex": "M",
    }
    result2 = predict_oa(clf, healthy_like_patient)
    print(f"  Healthy-like patient -> {result2}")
    assert result2["oa_probability"] < result["oa_probability"], \
        "Healthy-like patient should score a lower OA probability than OA-like patient"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  PASSED\n")
    print(f"All {len(tests)} classification tests passed.")
