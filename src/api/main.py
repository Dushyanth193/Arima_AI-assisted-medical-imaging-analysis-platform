"""
FastAPI Backend Application
===========================
Ties together every stage of the pipeline into clean REST endpoints:

    GET  /health
    GET  /implant-catalog
    POST /patients/{patient_reference}/process
    GET  /patients/{patient_reference}/history
    GET  /patients/{patient_reference}/mesh/{bone}
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse, ImplantComponentResponse, SizingResultResponse
from src.database.models import ImplantComponent, PatientMeasurement, get_session, init_db
from src.pipeline.inference_pipeline import run_pipeline_for_file, PipelineExecutionError
from src.utils.config import INFERENCE_OUTPUT_DIR

app = FastAPI(
    title="Knee Implant Sizing API",
    description="AI-assisted femoral/tibial measurement and implant size matching (decision-support only).",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/implant-catalog", response_model=list[ImplantComponentResponse])
def list_catalog(session: Session = Depends(get_session)) -> list[dict]:
    components = session.query(ImplantComponent).all()
    return [
        {
            "id": c.id,
            "manufacturer": c.manufacturer,
            "system_name": c.system_name,
            "component_type": c.component_type.value,
            "size_label": c.size_label,
            "ml_width_mm": c.ml_width_mm,
            "ap_dimension_mm": c.ap_dimension_mm,
            "tolerance_mm": c.tolerance_mm,
        }
        for c in components
    ]


@app.post("/patients/{patient_reference}/process", response_model=SizingResultResponse)
async def process_patient_ct(
    patient_reference: str,
    file: UploadFile = File(..., description="Knee CT as a .nii/.nii.gz file or .zip archive of DICOM slices"),
    manufacturer: str | None = None,
    system_name: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """
    End-to-end pipeline for one patient CT upload:
        Load -> Preprocess -> Segment -> Postprocess -> Measure ->
        Match Implants -> Reconstruct 3D Mesh -> Save & Return.
    """
    filename_lower = file.filename.lower()
    if not filename_lower.endswith((".nii", ".nii.gz", ".zip", ".gz")):
        raise HTTPException(400, "Supported formats: NIfTI (.nii, .nii.gz) or ZIP archive (.zip) of DICOM slices.")


    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = run_pipeline_for_file(
                input_path=tmp_path,
                patient_reference=patient_reference,
                db_session=session,
                manufacturer_filter=manufacturer,
                system_filter=system_name,
            )
            return result
        except PipelineExecutionError as e:
            raise HTTPException(500, f"Pipeline execution failed: {e}") from e
        except Exception as e:
            raise HTTPException(500, f"Unexpected error during processing: {e}") from e


@app.post("/patients/{patient_reference}/process_local_dicom", response_model=SizingResultResponse)
def process_local_dicom(
    patient_reference: str,
    dicom_path: str = "Patient_Dataset/Patient1",
    manufacturer: str | None = None,
    system_name: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """
    End-to-end pipeline execution on a local DICOM dataset directory (e.g. Patient_Dataset/Patient1).
    """

    d_path = Path(dicom_path)
    if not d_path.exists() or not d_path.is_dir():
        raise HTTPException(404, f"DICOM directory not found: {d_path}")

    try:
        result = run_pipeline_for_file(
            input_path=d_path,
            patient_reference=patient_reference,
            db_session=session,
            manufacturer_filter=manufacturer,
            system_filter=system_name,
        )
        return result
    except PipelineExecutionError as e:
        raise HTTPException(500, f"Pipeline execution failed: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during processing: {e}") from e


@app.get("/patients/{patient_reference}/history")
def get_patient_history(patient_reference: str, session: Session = Depends(get_session)) -> list[dict]:
    records = (
        session.query(PatientMeasurement)
        .filter(PatientMeasurement.patient_reference == patient_reference)
        .order_by(PatientMeasurement.created_at.desc())
        .all()
    )
    if not records:
        raise HTTPException(404, f"No history found for patient_reference={patient_reference}")

    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "femur_ml_width_mm": r.femur_ml_width_mm,
            "femur_ap_dimension_mm": r.femur_ap_dimension_mm,
            "tibia_ml_width_mm": r.tibia_ml_width_mm,
            "tibia_ap_dimension_mm": r.tibia_ap_dimension_mm,
            "matching_score_femoral": r.matching_score_femoral,
            "matching_score_tibial": r.matching_score_tibial,
        }
        for r in records
    ]


@app.get("/patients/{patient_reference}/mesh/{bone}")
def get_patient_mesh(patient_reference: str, bone: str):
    if bone not in ("femur", "tibia"):
        raise HTTPException(400, "bone parameter must be 'femur' or 'tibia'")

    mesh_file = INFERENCE_OUTPUT_DIR / patient_reference / f"{bone}.vtp"
    if not mesh_file.exists():
        raise HTTPException(404, f"3D Mesh for patient={patient_reference}, bone={bone} not found.")

    return FileResponse(str(mesh_file), media_type="application/octet-stream", filename=f"{patient_reference}_{bone}.vtp")
