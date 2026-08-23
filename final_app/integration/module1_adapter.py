"""
module1_adapter.py
------------------
Adapter interface connecting NEXORA Unified Platform to Module 1
(MRI Meniscus Segmentation & Osteoarthritis Diagnostics).

Calls the real functions in module1_mri/src in an isolated execution context.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE1_ROOT = PROJECT_ROOT / "module1_mri"


class Module1Context:
    """Isolates module1_mri imports and namespace from module-2."""
    def __enter__(self):
        self.old_path = list(sys.path)
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sys.modules.pop(k, None)
        if str(MODULE1_ROOT) in sys.path:
            sys.path.remove(str(MODULE1_ROOT))
        sys.path.insert(0, str(MODULE1_ROOT))
        return self

    def __exit__(self, *args):
        sys.path = self.old_path
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sys.modules.pop(k, None)


def get_available_models() -> Dict[str, Path]:
    """Returns available trained model checkpoint paths."""
    models_dir = MODULE1_ROOT / "models"
    available = {}
    
    calibrated_path = models_dir / "oa_classifier.joblib"
    if calibrated_path.exists():
        available["Calibrated Anatomical Model (Standard)"] = calibrated_path
        
    real_path = models_dir / "oa_classifier_real.joblib"
    if real_path.exists():
        available["Experimental RSNA Cohort Model"] = real_path
        
    return available


def load_module1_classifier(model_choice: Optional[str] = None):
    """Loads the requested or best available trained OA classifier."""
    available = get_available_models()
    if not available:
        return None, "No Model Found"
        
    if model_choice and model_choice in available:
        target_path = available[model_choice]
        target_name = model_choice
    else:
        target_name = list(available.keys())[0]
        target_path = available[target_name]

    with Module1Context():
        from src.classification import load_classifier
        clf = load_classifier(str(target_path))
        return clf, target_name


def get_module1_demo_scans() -> Dict[str, Path]:
    """Discovers available demo MRI scans across new_patient, reference_db, and real RSNA folders."""
    scans: Dict[str, Path] = {}
    
    # 1. Clinical Cases in new_patient
    new_pat_dir = MODULE1_ROOT / "data" / "new_patient"
    if new_pat_dir.exists():
        for f in sorted(new_pat_dir.glob("*.nii.gz")):
            if not f.name.endswith("_seg.nii.gz"):
                label = f"Clinical Demo: {f.name}"
                scans[label] = f
                
    # 2. Reference Cohort
    ref_db_dir = MODULE1_ROOT / "data" / "reference_db" / "images"
    if ref_db_dir.exists():
        for fname in ["REF_SUB_011.nii.gz", "REF_SUB_001.nii.gz", "REF_SUB_002.nii.gz", "REF_SUB_004.nii.gz"]:
            f = ref_db_dir / fname
            if f.exists():
                scans[f"Reference Cohort: {f.name}"] = f
                
    # 3. RSNA Cohort Samples
    rsna_dir = MODULE1_ROOT / "data" / "reference_db_real" / "images"
    if rsna_dir.exists():
        for f in sorted(rsna_dir.glob("*.nii.gz"))[:4]:
            if not f.name.endswith("_seg.nii.gz"):
                scans[f"RSNA Cohort Scan: {f.name}"] = f
                
    return scans


def load_mri_scan(file_obj_or_path: Any, filename: str) -> Dict[str, Any]:
    """Loads a 3D MRI from file path, uploaded bytes, or .npz array."""
    import SimpleITK as sitk
    import numpy as np

    if filename.endswith(".npz"):
        if hasattr(file_obj_or_path, "read"):
            data = np.load(file_obj_or_path)
        else:
            data = np.load(str(file_obj_or_path))
        array = data["data"] if "data" in data else list(data.values())[0]
        array_f = array.astype(np.float32)
        sitk_img = sitk.GetImageFromArray(array_f)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        return {
            "array": array_f,
            "spacing": (1.0, 1.0, 1.0),
            "sitk_image": sitk_img,
            "filename": filename,
        }
    else:
        with Module1Context():
            from src.io_utils import load_mri
            if hasattr(file_obj_or_path, "read"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                    tmp.write(file_obj_or_path.read())
                    tmp_path = tmp.name
                raw = load_mri(tmp_path)
                raw["filename"] = filename
                return raw
            else:
                raw = load_mri(str(file_obj_or_path))
                raw["filename"] = filename
                return raw


def process_mri_pipeline(
    raw_mri_dict: Dict[str, Any],
    age: float,
    sex: str,
    bmi: float,
    clf: Any,
) -> Dict[str, Any]:
    """Executes full 3D preprocessing, meniscus segmentation, landmark extrusion calculation, and OA prediction."""
    try:
        with Module1Context():
            from src.preprocessing import preprocess_pipeline
            from src.segmentation import segment_meniscus_placeholder
            from src.landmarks import detect_tibial_plateau_landmarks, estimate_tibial_plateau_mask
            from src.features import extract_features
            from src.classification import predict_oa
            from src.reporting import build_report

            sitk_img = raw_mri_dict["sitk_image"]
            preprocessed_img, norm_array = preprocess_pipeline(
                sitk_img,
                target_spacing=(0.5, 0.5, 0.5),
                bias_correct=True,
                denoise_flag=True,
            )
            
            meniscus_mask = segment_meniscus_placeholder(norm_array, intensity_percentile=95)
            spacing_zyx = preprocessed_img.GetSpacing()[::-1]
            
            landmark_res = detect_tibial_plateau_landmarks(norm_array, meniscus_mask, spacing_zyx)
            tibia_mask = landmark_res.get("tibia_mask")
            if tibia_mask is None:
                tibia_mask = estimate_tibial_plateau_mask(norm_array, meniscus_mask, spacing_zyx)
                
            features = extract_features(
                meniscus_mask,
                spacing_zyx,
                tibia_mask=tibia_mask,
                volume_array=norm_array,
            )
            
            features_for_model = {
                **features,
                "age": float(age),
                "sex": str(sex),
                "bmi": float(bmi),
            }
            
            if clf is not None:
                oa_result = predict_oa(clf, features_for_model)
            else:
                oa_result = {
                    "oa_classification": "Model Not Available",
                    "oa_probability": 0.0,
                }
                
            report = build_report(
                patient_id=raw_mri_dict.get("filename", "UNKNOWN_PATIENT"),
                features=features,
                oa_result=oa_result,
            )
            
            return {
                "success": True,
                "raw": raw_mri_dict,
                "preprocessed_img": preprocessed_img,
                "norm_array": norm_array,
                "meniscus_mask": meniscus_mask,
                "tibia_mask": tibia_mask,
                "landmark_res": landmark_res,
                "features": features,
                "oa_result": oa_result,
                "report": report,
                "spacing_zyx": spacing_zyx,
                "patient_info": {"age": age, "sex": sex, "bmi": bmi},
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
