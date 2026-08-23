"""
Inference CLI Script
Runs full pipeline on a single CT scan file/dir from command line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.inference_pipeline import run_pipeline_for_file



def main():
    parser = argparse.ArgumentParser(description="Run full end-to-end knee sizing pipeline on a scan.")
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI file or DICOM dir")
    parser.add_argument("--patient-id", type=str, default="cli-patient-001", help="Patient Reference ID")
    parser.add_argument("--manufacturer", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    args = parser.parse_args()

    print(f"Running pipeline for input: {args.input}")
    res = run_pipeline_for_file(
        input_path=args.input,
        patient_reference=args.patient_id,
        manufacturer_filter=args.manufacturer,
        system_filter=args.system,
    )
    print("\n--- Pipeline Result Summary ---")
    print(json.dumps({
        "patient_reference": res["patient_reference"],
        "measurements": res["measurements"],
        "recommended_femur_size": res["femoral"]["recommended"]["size_label"] if res["femoral"]["recommended"] else None,
        "recommended_tibia_size": res["tibial"]["recommended"]["size_label"] if res["tibial"]["recommended"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
