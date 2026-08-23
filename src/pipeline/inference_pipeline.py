"""
Orchestrated Inference Pipeline Service
=======================================
Implements the single end-to-end execution pipeline connecting all stages:
Image Loading -> Preprocessing -> 3D Segmentation -> Mask Postprocessing ->
Anatomical Measurements -> 3D Mesh Reconstruction -> Database Sizing Query ->
Implant Candidate Matching & Ranking.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import SimpleITK as sitk
from sqlalchemy.orm import Session

from src.io.dicom_loader import load_dicom_series
from src.io.nifti_loader import load_nifti_volume
from src.io.metadata import extract_image_metadata
from src.preprocessing.ct_preprocessing import preprocess_ct
from src.segmentation.infer import segment_ct
from src.postprocessing.mask_cleaning import clean_multiclass_mask
from src.measurement.anatomical_measurement import extract_anatomical_measurements, measurements_to_dict
from src.reconstruction.mesh_reconstruction import reconstruct_bones, save_meshes
from src.matching.implant_matcher import match_patient_to_implants
from src.database.models import PatientMeasurement, get_session, init_db
from src.utils.config import INFERENCE_OUTPUT_DIR, LABELS


class PipelineExecutionError(Exception):
    """Raised when an error occurs during end-to-end pipeline execution."""


def run_pipeline_for_file(
    input_path: str | Path,
    patient_reference: str,
    db_session: Session | None = None,
    manufacturer_filter: str | None = None,
    system_filter: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executes the complete research pipeline on a single CT/MRI scan input file or directory.

    Parameters
    ----------
    input_path : str | Path
        Path to a .nii/.nii.gz file or DICOM directory.
    patient_reference : str
        De-identified case / patient reference ID.
    db_session : Session, optional
        SQLAlchemy database session.
    manufacturer_filter : str, optional
        Filter catalog matching by specific manufacturer.
    system_filter : str, optional
        Filter catalog matching by specific system name.
    checkpoint_path : str | Path, optional
        Path to trained MONAI segmentation model checkpoint.

    Returns
    -------
    dict
        Structured result including metadata, anatomical measurements,
        implant recommendations, candidate rankings, and mesh output paths.
    """
    import zipfile

    input_path = Path(input_path)
    if not input_path.exists():
        raise PipelineExecutionError(f"Input path does not exist: {input_path}")

    # Handle zip archive inputs
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = input_path.parent / f"extracted_{input_path.stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        input_path = extract_dir

    # 1. Load image and extract spatial metadata
    if input_path.is_dir():
        reader = sitk.ImageSeriesReader()
        dicom_dirs = [d for d in [input_path] + list(input_path.rglob("*")) if d.is_dir() and reader.GetGDCMSeriesIDs(str(d))]
        nii_files = list(input_path.rglob("*.nii")) + list(input_path.rglob("*.nii.gz"))
        
        if dicom_dirs:
            image, meta = load_dicom_series(dicom_dirs[0])
            input_path = dicom_dirs[0]
        elif nii_files:
            image, meta = load_nifti_volume(nii_files[0])
            input_path = nii_files[0]
        else:
            raise PipelineExecutionError(f"No DICOM series or NIfTI file found in directory: {input_path}")
    else:
        image, meta = load_nifti_volume(input_path)


    # 2. Run segmentation (preprocesses internally via SimpleITK)
    label_mask, preprocessed_image = segment_ct(
        str(input_path), checkpoint_path=checkpoint_path
    )

    # 3. Postprocess mask (connected component cleaning & hole filling)
    mask_array = sitk.GetArrayFromImage(label_mask)
    cleaned_array = clean_multiclass_mask(mask_array, num_classes=len(LABELS))
    cleaned_label_mask = sitk.GetImageFromArray(cleaned_array)
    cleaned_label_mask.CopyInformation(label_mask)

    # 4. Extract physical anatomical measurements (mm / mm3)
    measurements = extract_anatomical_measurements(cleaned_label_mask)
    meas_dict = measurements_to_dict(measurements)

    # 5. Generate 3D surface meshes (VTK/PyVista)
    meshes = reconstruct_bones(cleaned_label_mask)
    patient_output_dir = INFERENCE_OUTPUT_DIR / patient_reference
    save_meshes(meshes, patient_output_dir, file_format="vtp")

    # 6. Database Implant Matching & Candidate Ranking
    close_session_after = False
    if db_session is None:
        db_generator = get_session()
        db_session = next(db_generator)
        close_session_after = True

    try:
        matching_results = match_patient_to_implants(
            session=db_session,
            femur_ml_mm=measurements["femur"].ml_width_mm,
            femur_ap_mm=measurements["femur"].ap_dimension_mm,
            tibia_ml_mm=measurements["tibia"].ml_width_mm,
            tibia_ap_mm=measurements["tibia"].ap_dimension_mm,
            manufacturer=manufacturer_filter,
            system_name=system_filter,
        )

        # 7. Persist measurement and top match to database
        fem_rec = matching_results["femoral"]["recommended"]
        tib_rec = matching_results["tibial"]["recommended"]

        pm = PatientMeasurement(
            patient_reference=patient_reference,
            femur_ml_width_mm=measurements["femur"].ml_width_mm,
            femur_ap_dimension_mm=measurements["femur"].ap_dimension_mm,
            tibia_ml_width_mm=measurements["tibia"].ml_width_mm,
            tibia_ap_dimension_mm=measurements["tibia"].ap_dimension_mm,
            recommended_femoral_component_id=fem_rec.component_id if fem_rec else None,
            recommended_tibial_component_id=tib_rec.component_id if tib_rec else None,
            matching_score_femoral=fem_rec.matching_score if fem_rec else None,
            matching_score_tibial=tib_rec.matching_score if tib_rec else None,
        )
        db_session.add(pm)
        db_session.commit()
    finally:
        if close_session_after:
            db_session.close()

    def _serialize_comp(rec_dict: dict) -> dict:
        rec = rec_dict.get("recommended")
        alts = rec_dict.get("alternatives", [])
        return {
            "recommended": vars(rec) if rec else None,
            "alternatives": [vars(a) for a in alts],
            "confidence": rec_dict.get("confidence", "moderate"),
        }

    return {
        "disclaimer": "Research / prototype result — not for clinical decision-making. Final implant selection remains with a qualified orthopedic surgeon.",
        "patient_reference": patient_reference,
        "image_metadata": meta.to_dict(),
        "measurements": meas_dict,
        "femoral": _serialize_comp(matching_results["femoral"]),
        "tibial": _serialize_comp(matching_results["tibial"]),
        "mesh_paths": {
            "femur": str(patient_output_dir / "femur.vtp"),
            "tibia": str(patient_output_dir / "tibia.vtp"),
        },
    }
