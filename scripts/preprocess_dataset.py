"""
Preprocess Dataset CLI Script
Runs SimpleITK CT preprocessing (QC, LPS reorientation, curvature flow denoising,
1.0mm isotropic resampling, HU clipping) on raw DICOM (e.g. RTKN0668_CT_intact) or NIfTI scans.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ct_preprocessing import preprocess_ct
from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PATIENT_DATASET_DIR




def generate_baseline_bone_label(processed_img: sitk.Image) -> sitk.Image:
    """
    Generate a baseline label mask (1=femur, 2=tibia, 0=background)
    from a normalized preprocessed CT volume via intensity thresholding and spatial centroids.
    """
    arr = sitk.GetArrayFromImage(processed_img)  # (D, H, W)
    bone_binary = arr >= 0.20  # ~250+ HU in normalized volume

    labeled, num_features = label(bone_binary)
    if num_features == 0:
        bone_binary = arr >= 0.15
        labeled, num_features = label(bone_binary)

    out_mask = np.zeros_like(arr, dtype=np.uint8)
    if num_features > 0:
        component_info = []
        for i in range(1, num_features + 1):
            mask_i = (labeled == i)
            size = int(np.sum(mask_i))
            if size > 500:
                z_c = float(center_of_mass(mask_i)[0])
                component_info.append((size, z_c, i))

        component_info.sort(key=lambda x: x[0], reverse=True)
        top_comps = component_info[:2]

        if len(top_comps) == 1:
            size, z_c, comp_id = top_comps[0]
            mask_comp = (labeled == comp_id)
            z_indices = np.where(mask_comp)[0]
            z_mid = (np.min(z_indices) + np.max(z_indices)) / 2.0
            z_coords = np.arange(arr.shape[0])[:, None, None]
            out_mask[mask_comp & (z_coords >= z_mid)] = 1
            out_mask[mask_comp & (z_coords < z_mid)] = 2
        elif len(top_comps) >= 2:
            top_comps.sort(key=lambda x: x[1], reverse=True)
            femur_comp_id = top_comps[0][2]
            tibia_comp_id = top_comps[1][2]
            out_mask[labeled == femur_comp_id] = 1
            out_mask[labeled == tibia_comp_id] = 2

    label_img = sitk.GetImageFromArray(out_mask)
    label_img.CopyInformation(processed_img)
    return label_img


def main():
    parser = argparse.ArgumentParser(description="Bulk preprocess raw CT volumes.")
    parser.add_argument("--raw-dir", type=str, default=str(DATA_RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DATA_PROCESSED_DIR))
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    items = []
    # Check for Patient_Dataset in project root
    if PATIENT_DATASET_DIR.exists() and PATIENT_DATASET_DIR.is_dir():
        patient_dirs = [d for d in PATIENT_DATASET_DIR.iterdir() if d.is_dir()]
        items.extend(patient_dirs)

    # Check raw_dir
    if raw_path.exists():
        items.extend(list(raw_path.glob("*.nii*")) + [d for d in raw_path.iterdir() if d.is_dir()])

    print(f"Found {len(items)} scan dataset items to preprocess.", flush=True)


    for item in items:
        case_id = item.name.split(".")[0]
        case_out = out_path / case_id
        case_out.mkdir(exist_ok=True)

        try:
            print(f"Preprocessing DICOM/NIfTI volume: {item}...", flush=True)
            processed = preprocess_ct(item)
            out_file = case_out / "image.nii.gz"
            sitk.WriteImage(processed, str(out_file))

            # Generate label mask for training/testing dataloaders
            print(f"Generating baseline bone label mask for {case_id}...", flush=True)
            label_img = generate_baseline_bone_label(processed)
            label_file = case_out / "label.nii.gz"
            sitk.WriteImage(label_img, str(label_file))

            print(f"[OK] Successfully preprocessed {case_id} -> {out_file}", flush=True)
            print(f"   Saved image and label mask in {case_out}", flush=True)
            print(f"   Dimensions: {processed.GetSize()}, Spacing: {processed.GetSpacing()}", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed preprocessing for {item}: {e}", flush=True)




if __name__ == "__main__":
    main()

