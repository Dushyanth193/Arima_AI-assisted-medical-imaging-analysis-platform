"""
module2_adapter.py
------------------
Adapter interface connecting NEXORA Unified Platform to Module 2
(CT Femur/Tibia Segmentation, Anatomical Sizing & Implant Matching).

Calls the real functions in module-2/src in an isolated execution context.
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE2_ROOT = PROJECT_ROOT / "module-2"

# Configure database URL to existing SQLite database in module-2
DB_PATH = MODULE2_ROOT / "knee_implant.db"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"


class Module2Context:
    """Isolates module-2 imports and namespace from module1_mri."""
    def __enter__(self):
        self.old_path = list(sys.path)
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sys.modules.pop(k, None)
        if str(MODULE2_ROOT) in sys.path:
            sys.path.remove(str(MODULE2_ROOT))
        sys.path.insert(0, str(MODULE2_ROOT))
        os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
        return self

    def __exit__(self, *args):
        sys.path = self.old_path
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sys.modules.pop(k, None)


def get_module2_demo_patients() -> Dict[str, Path]:
    """Returns available preloaded patient CT DICOM directories."""
    dataset_dir = MODULE2_ROOT / "Patient_Dataset"
    patients: Dict[str, Path] = {}
    if dataset_dir.exists():
        for p_dir in sorted(dataset_dir.iterdir()):
            if p_dir.is_dir():
                dcm_files = list(p_dir.glob("*.DCM*")) + list(p_dir.glob("*.dcm"))
                count = len(dcm_files)
                label = f"CT Patient Case: {p_dir.name} ({count} DICOM slices)"
                patients[label] = p_dir
    return patients


def get_catalog_summary() -> Dict[str, Any]:
    """Queries the implant database to retrieve available manufacturers and size counts."""
    try:
        with Module2Context():
            from src.database.models import init_db, get_session, ImplantComponent
            init_db()
            db_gen = get_session()
            session = next(db_gen)
            try:
                components = session.query(ImplantComponent).all()
                total = len(components)
                manufacturers = sorted(list(set(c.manufacturer for c in components)))
                systems = sorted(list(set(c.system_name for c in components)))
                return {
                    "available": True,
                    "total_components": total,
                    "manufacturers": manufacturers,
                    "systems": systems,
                    "db_path": str(DB_PATH),
                }
            finally:
                session.close()
    except Exception as e:
        return {"available": False, "error": str(e), "total_components": 0, "manufacturers": []}


def get_checkpoint_status() -> Dict[str, Any]:
    """Verifies existence of trained MONAI DynUNet segmentation weights."""
    best_pt = MODULE2_ROOT / "checkpoints" / "best_model.pt"
    final_pt = MODULE2_ROOT / "checkpoints" / "final_model.pt"
    
    return {
        "best_exists": best_pt.exists(),
        "final_exists": final_pt.exists(),
        "path": str(best_pt if best_pt.exists() else final_pt),
    }


def process_ct_pipeline(
    input_source: Any,
    filename_or_ref: str,
    patient_ref: str = "PATIENT_CT_001",
    manufacturer_filter: Optional[str] = None,
    system_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes Module 2 end-to-end pipeline:
    CT Loading -> 3D Preprocessing -> MONAI DynUNet Bone Segmentation ->
    Morphometric Sizing (AP/ML) -> Implant Sizing Query & Candidate Matching.
    """
    try:
        # Determine actual input path (file path, directory path, or uploaded buffer)
        temp_dir = None
        if isinstance(input_source, (str, Path)):
            input_path = Path(input_source)
        else:
            temp_dir = tempfile.mkdtemp(prefix="nexora_ct_upload_")
            if filename_or_ref.endswith(".zip"):
                zip_path = Path(temp_dir) / filename_or_ref
                with open(zip_path, "wb") as f:
                    f.write(input_source.getvalue())
                input_path = zip_path
            else:
                ext = ".nii.gz" if filename_or_ref.endswith(".nii.gz") else Path(filename_or_ref).suffix
                target_file = Path(temp_dir) / f"upload{ext}"
                with open(target_file, "wb") as f:
                    f.write(input_source.getvalue())
                input_path = target_file

        ckpt_status = get_checkpoint_status()
        ckpt_path = ckpt_status["path"] if (ckpt_status["best_exists"] or ckpt_status["final_exists"]) else None

        with Module2Context():
            from src.pipeline.inference_pipeline import run_pipeline_for_file
            result = run_pipeline_for_file(
                input_path=input_path,
                patient_reference=patient_ref,
                manufacturer_filter=manufacturer_filter if manufacturer_filter != "All" else None,
                system_filter=system_filter if system_filter != "All" else None,
                checkpoint_path=ckpt_path,
            )

            return {
                "success": True,
                "patient_reference": patient_ref,
                "metadata": result.get("image_metadata", {}),
                "measurements": result.get("measurements", {}),
                "femoral": result.get("femoral", {}),
                "tibial": result.get("tibial", {}),
                "mesh_paths": result.get("mesh_paths", {}),
                "disclaimer": result.get("disclaimer", ""),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
