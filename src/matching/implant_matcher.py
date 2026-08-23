"""
Implant Size Matching Model
============================
Implements the flow diagram's "Implant Size Matching Model" box:
    - Compares patient measurements with the database
    - Compares anatomy with implant dimensions
    - Applies the selected implant-system standards
    - Identifies the closest matching component sizes
    - Calculates measurement-matching differences

Design choice (per earlier architecture discussion): this is a
deterministic geometric nearest-neighbor matcher, not a black-box deep
learning classifier. It operates on ML width + AP dimension, which is
exactly what the flow diagram's feature-extraction stage produces, and
its "distance" output maps directly onto the diagram's "measurement-
matching score" field - fully explainable to the reviewing surgeon.

An optional ML calibration layer (e.g. gradient-boosted trees trained on
real implanted-size outcomes) is left as a documented future extension
in Limitations & Solutions, since it requires historical outcome labels
this basic prototype does not have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance
from sqlalchemy.orm import Session

from src.database.models import ComponentType, ImplantComponent


@dataclass
class SizeCandidate:
    component_id: int
    size_label: str
    ml_width_mm: float
    ap_dimension_mm: float
    ml_diff_mm: float
    ap_diff_mm: float
    matching_score: float          # weighted Euclidean distance in mm; lower = better fit
    within_tolerance: bool
    overhang_risk: bool            # implant larger than bone on either axis beyond tolerance
    undercoverage_risk: bool       # implant smaller than bone on either axis beyond tolerance


def _score_candidate(
    patient_ml: float,
    patient_ap: float,
    component: ImplantComponent,
    ml_weight: float = 1.0,
    ap_weight: float = 1.0,
) -> SizeCandidate:
    ml_diff = component.ml_width_mm - patient_ml
    ap_diff = component.ap_dimension_mm - patient_ap

    # Weighted Euclidean distance in measurement space. Equal weighting
    # by default; ml_weight/ap_weight are exposed so this can be tuned
    # per implant system if one axis is clinically more forgiving than
    # the other (a documented, not-yet-calibrated extension point).
    score = float(
        distance.euclidean(
            [ml_weight * patient_ml, ap_weight * patient_ap],
            [ml_weight * component.ml_width_mm, ap_weight * component.ap_dimension_mm],
        )
    )

    within_tolerance = abs(ml_diff) <= component.tolerance_mm and abs(ap_diff) <= component.tolerance_mm
    overhang_risk = ml_diff > component.tolerance_mm or ap_diff > component.tolerance_mm
    undercoverage_risk = ml_diff < -component.tolerance_mm or ap_diff < -component.tolerance_mm

    return SizeCandidate(
        component_id=component.id,
        size_label=component.size_label,
        ml_width_mm=component.ml_width_mm,
        ap_dimension_mm=component.ap_dimension_mm,
        ml_diff_mm=round(ml_diff, 2),
        ap_diff_mm=round(ap_diff, 2),
        matching_score=round(score, 3),
        within_tolerance=within_tolerance,
        overhang_risk=overhang_risk,
        undercoverage_risk=undercoverage_risk,
    )


def rank_candidates(
    session: Session,
    patient_ml_mm: float,
    patient_ap_mm: float,
    component_type: ComponentType,
    manufacturer: str | None = None,
    system_name: str | None = None,
    top_k: int = 3,
) -> list[SizeCandidate]:
    """
    Query the implant catalog for the requested component type (and
    optionally a specific manufacturer/system), score every candidate
    size against the patient's measurements, and return the top_k
    closest matches ranked by matching_score ascending.
    """
    query = session.query(ImplantComponent).filter(ImplantComponent.component_type == component_type)
    if manufacturer:
        query = query.filter(ImplantComponent.manufacturer == manufacturer)
    if system_name:
        query = query.filter(ImplantComponent.system_name == system_name)

    components = query.all()
    if not components:
        raise ValueError(
            f"No implant catalog entries found for component_type={component_type}, "
            f"manufacturer={manufacturer}, system_name={system_name}. "
            f"Has the catalog been seeded? See src/database/seed_implant_catalog.py."
        )

    scored = [_score_candidate(patient_ml_mm, patient_ap_mm, c) for c in components]
    scored.sort(key=lambda c: c.matching_score)
    return scored[:top_k]


def match_patient_to_implants(
    session: Session,
    femur_ml_mm: float,
    femur_ap_mm: float,
    tibia_ml_mm: float,
    tibia_ap_mm: float,
    manufacturer: str | None = None,
    system_name: str | None = None,
    top_k: int = 3,
) -> dict:
    """
    Convenience wrapper matching both femoral and tibial components in
    one call - what the FastAPI endpoint and Streamlit UI actually use.
    """
    femoral_candidates = rank_candidates(
        session, femur_ml_mm, femur_ap_mm, ComponentType.FEMORAL, manufacturer, system_name, top_k,
    )
    tibial_candidates = rank_candidates(
        session, tibia_ml_mm, tibia_ap_mm, ComponentType.TIBIAL, manufacturer, system_name, top_k,
    )

    best_femoral = femoral_candidates[0]
    best_tibial = tibial_candidates[0]

    confidence_femoral = "high" if best_femoral.within_tolerance else "review_recommended"
    confidence_tibial = "high" if best_tibial.within_tolerance else "review_recommended"

    return {
        "femoral": {
            "recommended": best_femoral,
            "alternatives": femoral_candidates[1:],
            "confidence": confidence_femoral,
        },
        "tibial": {
            "recommended": best_tibial,
            "alternatives": tibial_candidates[1:],
            "confidence": confidence_tibial,
        },
    }
