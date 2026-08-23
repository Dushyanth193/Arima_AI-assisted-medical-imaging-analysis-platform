from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", example="ok")
    version: str = Field(default="0.1.0")


class ImplantComponentResponse(BaseModel):
    id: int
    manufacturer: str
    system_name: str
    component_type: str
    size_label: str
    ml_width_mm: float
    ap_dimension_mm: float
    tolerance_mm: float = 1.5


class MeasurementResponse(BaseModel):
    femur_ml_width_mm: float
    femur_ap_dimension_mm: float
    femur_volume_mm3: float
    tibia_ml_width_mm: float
    tibia_ap_dimension_mm: float
    tibia_volume_mm3: float


class SizeCandidateResponse(BaseModel):
    component_id: int
    size_label: str
    ml_width_mm: float
    ap_dimension_mm: float
    ml_diff_mm: float
    ap_diff_mm: float
    matching_score: float
    within_tolerance: bool
    overhang_risk: bool
    undercoverage_risk: bool


class ImplantRecommendationResponse(BaseModel):
    recommended: SizeCandidateResponse | None = None
    alternatives: list[SizeCandidateResponse] = []
    confidence: str = "moderate"


class SizingResultResponse(BaseModel):
    disclaimer: str
    patient_reference: str
    measurements: MeasurementResponse
    femoral: ImplantRecommendationResponse
    tibial: ImplantRecommendationResponse
