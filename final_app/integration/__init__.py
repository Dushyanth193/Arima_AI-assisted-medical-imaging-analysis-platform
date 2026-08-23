"""NEXORA Unified Integration Layer"""
from .module1_adapter import (
    load_mri_scan,
    process_mri_pipeline,
    load_module1_classifier,
    get_module1_demo_scans,
    get_available_models as get_module1_models,
)
from .module2_adapter import (
    process_ct_pipeline,
    get_module2_demo_patients,
    get_catalog_summary as get_module2_catalog,
    get_checkpoint_status as get_module2_checkpoint_status,
)

__all__ = [
    "load_mri_scan",
    "process_mri_pipeline",
    "load_module1_classifier",
    "get_module1_demo_scans",
    "get_module1_models",
    "process_ct_pipeline",
    "get_module2_demo_patients",
    "get_module2_catalog",
    "get_module2_checkpoint_status",
]
