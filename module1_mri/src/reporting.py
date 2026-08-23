"""
Turns raw features + classifier output into the structured result shown in
the diagram's "OA Assessment Output" box, plus a quick visual overlay.

The dict shape returned by build_report() is deliberately stable - this is
what the future "Integrated AI Report" (combining Module 1 + Module 2)
would consume, so avoid renaming keys casually once Module 2 exists.
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_segmentation_overlay(mri_slice: np.ndarray, mask_slice: np.ndarray, title="Meniscus Segmentation", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mri_slice, cmap="gray")
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(overlay, cmap="autumn", alpha=0.45)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def build_report(patient_id: str, features: dict, oa_result: dict) -> dict:
    return {
        "patient_id": patient_id,
        "oa_classification": oa_result["oa_classification"],
        "oa_probability_pct": round(oa_result["oa_probability"] * 100, 1),
        "meniscus_volume_cm3": round(features["meniscus_volume_cm3"], 2),
        "meniscus_thickness_mm": round(features["meniscus_thickness_mm"], 2),
        "meniscus_extrusion_mm": (
            round(features["meniscus_extrusion_mm"], 2)
            if features.get("meniscus_extrusion_mm") is not None
            else None
        ),
    }


def save_report_json(report: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def print_report(report: dict) -> None:
    print("OA Assessment Output")
    print(f"  OA Classification : {report['oa_classification']}")
    print(f"  OA Probability    : {report['oa_probability_pct']}%")
    print(f"  Meniscus Volume   : {report['meniscus_volume_cm3']} cm^3")
    print(f"  Meniscus Thickness: {report['meniscus_thickness_mm']} mm")
    print(f"  Meniscus Extrusion: {report['meniscus_extrusion_mm']} mm")
