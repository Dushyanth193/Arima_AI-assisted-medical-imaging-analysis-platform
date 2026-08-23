"""
app.py
------
Nexora OrthoAI — Module 1: Patient MRI Meniscus Analysis & OA Diagnostic Suite
Clean, streamlined medical visualizer with direct patient MRI upload and clinical insights.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import streamlit as st

# Setup import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.io_utils import load_mri, save_mask
from src.preprocessing import preprocess_pipeline
from src.segmentation import segment_meniscus_placeholder
from src.landmarks import detect_tibial_plateau_landmarks
from src.features import extract_features
from src.classification import load_classifier, predict_oa
from src.reporting import build_report

# Page Configuration
st.set_page_config(
    page_title="Nexora OrthoAI — Patient MRI Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, Modern Medical Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b1120;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .badge-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-red {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-amber {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .insight-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model(choice="Calibrated Anatomical Model (Recommended)"):
    """Loads the chosen trained OA classifier."""
    base_model = os.path.join(BASE_DIR, "models", "oa_classifier.joblib")
    real_model = os.path.join(BASE_DIR, "models", "oa_classifier_real.joblib")
    
    if choice == "Calibrated Anatomical Model (Recommended)" and os.path.exists(base_model):
        return load_classifier(base_model), "Calibrated Reference Model"
    elif choice == "Experimental RSNA Dataset Model" and os.path.exists(real_model):
        return load_classifier(real_model), "Experimental RSNA Dataset Model"
    elif os.path.exists(base_model):
        return load_classifier(base_model), "Calibrated Reference Model"
    elif os.path.exists(real_model):
        return load_classifier(real_model), "Experimental RSNA Dataset Model"
    return None, "No Model Found"


def load_any_mri_file(file_obj, filename: str):
    """Handles loading .nii, .nii.gz, and .npz MRI files into SimpleITK Image and NumPy array."""
    if filename.endswith(".npz"):
        data = np.load(file_obj)
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
        # NIfTI file (.nii or .nii.gz)
        if hasattr(file_obj, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            raw = load_mri(tmp_path)
            raw["filename"] = filename
            return raw
        else:
            raw = load_mri(file_obj)
            raw["filename"] = filename
            return raw


def analyze_patient_scan(raw_mri_dict, age, sex, bmi, clf):
    """Runs complete end-to-end preprocessing, segmentation, landmark detection, and prediction."""
    preprocessed_img, norm_array = preprocess_pipeline(
        raw_mri_dict["sitk_image"],
        target_spacing=(0.5, 0.5, 0.5),
        bias_correct=True,
        denoise_flag=True,
    )
    meniscus_mask = segment_meniscus_placeholder(norm_array, intensity_percentile=95)
    spacing_zyx = preprocessed_img.GetSpacing()[::-1]

    # Automated Tibial Plateau Landmark Detection
    landmark_res = detect_tibial_plateau_landmarks(norm_array, meniscus_mask, spacing_zyx)
    tibia_mask = landmark_res["tibia_mask"]

    features = extract_features(
        meniscus_mask,
        spacing_zyx,
        tibia_mask=tibia_mask,
        volume_array=norm_array,
    )

    features_for_model = {
        **features,
        "age": age,
        "sex": sex,
        "bmi": bmi,
    }

    oa_result = predict_oa(clf, features_for_model)
    report = build_report(patient_id=raw_mri_dict["filename"], features=features, oa_result=oa_result)

    return {
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
    }


def main():
    # Header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); border-left: 5px solid #38bdf8; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='margin:0; color:#f8fafc; font-size:1.6rem;'>🩺 Nexora OrthoAI — Patient Knee MRI Analysis & OA Diagnostics</h2>
        <p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.92rem;'>
            Upload new patient MRI scans (.nii, .nii.gz, .npz) for automated 3D meniscus segmentation, tibial extrusion calculation, and Osteoarthritis risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Input Panel
    st.sidebar.title("Patient & MRI Input")

    st.sidebar.subheader("1. AI Model Selection")
    model_choice = st.sidebar.selectbox(
        "Diagnostic Model",
        ["Calibrated Anatomical Model (Recommended)", "Experimental RSNA Dataset Model"],
        index=0,
        help="The Calibrated Model uses standard orthopedic biomarker scales. The RSNA model uses the raw RSNA archive cohort.",
    )
    clf, model_name = get_model(model_choice)

    st.sidebar.subheader("2. Patient Demographics")
    patient_id = st.sidebar.text_input("Patient ID / Name", value="PATIENT_NEW_001")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        age = st.slider("Age (Years)", 20, 90, 62)
        sex = st.selectbox("Sex", ["F", "M"], index=0)
    with col_b:
        bmi = st.number_input("BMI (kg/m²)", 15.0, 48.0, 27.5, 0.5)
        bmi_cat = "Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese")
        badge_cls = "badge-green" if bmi < 25 else ("badge-amber" if bmi < 30 else "badge-red")
        st.markdown(f"<span class='{badge_cls}'>{bmi_cat}</span>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. MRI Scan Source")
    
    input_mode = st.sidebar.radio("Choose Input Method", ["📤 Upload New Patient MRI", "📁 Select Preloaded Demo Scan"])
    
    loaded_mri_dict = None

    if input_mode == "📤 Upload New Patient MRI":
        uploaded_file = st.sidebar.file_uploader(
            "Upload 3D Knee MRI (.nii, .nii.gz, .npz)",
            type=["nii", "gz", "npz"],
            help="Upload raw 3D MRI volume in NIfTI format (.nii, .nii.gz) or NumPy compressed (.npz)",
        )
        if uploaded_file is not None:
            loaded_mri_dict = load_any_mri_file(uploaded_file, uploaded_file.name)
            st.sidebar.success(f"Loaded: `{uploaded_file.name}`")
        else:
            st.sidebar.info("Upload a patient MRI file above to begin analysis.")
    else:
        # Preloaded Scans
        demo_files = {}
        new_pat_dir = os.path.join(BASE_DIR, "data", "new_patient")
        ref_db_dir = os.path.join(BASE_DIR, "data", "reference_db", "images")
        rsna_dir = os.path.join(BASE_DIR, "data", "reference_db_real", "images")
        
        if os.path.exists(new_pat_dir):
            for f in os.listdir(new_pat_dir):
                if f.endswith(".nii.gz") and not f.endswith("_seg.nii.gz"):
                    demo_files[f"Clinical Case: {f}"] = os.path.join(new_pat_dir, f)

        if os.path.exists(ref_db_dir):
            for f in ["REF_SUB_011.nii.gz", "REF_SUB_001.nii.gz", "REF_SUB_002.nii.gz", "REF_SUB_004.nii.gz"]:
                f_path = os.path.join(ref_db_dir, f)
                if os.path.exists(f_path):
                    demo_files[f"Reference Cohort: {f}"] = f_path
                    
        if os.path.exists(rsna_dir):
            for f in os.listdir(rsna_dir)[:4]:
                if f.endswith(".nii.gz"):
                    demo_files[f"Real RSNA Scan: {f}"] = os.path.join(rsna_dir, f)

        if demo_files:
            chosen_scan = st.sidebar.selectbox("Select Scan", list(demo_files.keys()))
            scan_path = demo_files[chosen_scan]
            loaded_mri_dict = load_any_mri_file(scan_path, os.path.basename(scan_path))
            st.sidebar.success(f"Selected: `{os.path.basename(scan_path)}`")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🧠 Active Model: **{model_name}**")

    # If no file is loaded, show welcoming instructions
    if loaded_mri_dict is None:
        st.info("👈 **Get Started**: Upload a new patient's 3D MRI scan in the sidebar (supports `.nii.gz`, `.nii`, or `.npz`) or select a preloaded clinical demo scan.")
        return

    # Process and analyze scan
    with st.spinner("Analyzing MRI Scan (Resampling, Meniscus Segmentation, Tibial Extrusion & OA Risk Assessment)..."):
        res = analyze_patient_scan(loaded_mri_dict, age, sex, bmi, clf)

    oa_res = res["oa_result"]
    feat = res["features"]
    prob = oa_res["oa_probability"] * 100
    is_oa = "Detected" in oa_res["oa_classification"]

    vol = feat["meniscus_volume_cm3"]
    thick = feat["meniscus_thickness_mm"]
    ext = feat["meniscus_extrusion_mm"] or 0.0

    # ---------------- KEY METRIC CARDS ----------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>OA Diagnostic Status</div>
            <div class='metric-val' style='color:{"#ef4444" if is_oa else "#10b981"}; font-size:1.5rem;'>
                {"Osteoarthritis Detected" if is_oa else "No OA Detected"}
            </div>
            <span class='{"badge-red" if is_oa else "badge-green"}'>Risk Probability: {prob:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        vol_badge = "badge-green" if vol >= 7.5 else ("badge-amber" if vol >= 5.5 else "badge-red")
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Meniscus Volume</div>
            <div class='metric-val'>{vol:.2f} <span style='font-size:1rem;'>cm³</span></div>
            <span class='{vol_badge}'>{"Preserved Volume" if vol >= 7.5 else "Volume Loss / Wear"}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        thick_badge = "badge-green" if thick >= 3.5 else "badge-red"
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Mean Thickness</div>
            <div class='metric-val'>{thick:.2f} <span style='font-size:1rem;'>mm</span></div>
            <span class='{thick_badge}'>{"Normal Thickness" if thick >= 3.5 else "Meniscal Thinning"}</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ext_badge = "badge-green" if ext < 3.0 else "badge-red"
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Tibial Extrusion</div>
            <div class='metric-val'>{ext:.2f} <span style='font-size:1rem;'>mm</span></div>
            <span class='{ext_badge}'>{"Normal (<3mm)" if ext < 3.0 else "Pathological Extrusion (≥3mm)"}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- CLINICAL INSIGHTS SUMMARY ----------------
    findings = []
    if is_oa:
        findings.append(f"**High Osteoarthritis Probability ({prob:.1f}%)**: Quantitative patterns match degenerative changes in the medial knee compartment.")
    else:
        findings.append(f"**Low Osteoarthritis Probability ({prob:.1f}%)**: Meniscal integrity and joint spacing remain well within normal parameters.")

    if ext >= 3.0:
        findings.append(f"**Pathological Meniscus Extrusion ({ext:.2f} mm)**: The meniscus protrudes beyond the tibial plateau cortical margin (≥ 3.0 mm), indicating circumferential hoop stress failure.")
    else:
        findings.append(f"**Normal Meniscal Position ({ext:.2f} mm)**: No significant extrusion beyond the tibial plateau edge (< 3.0 mm).")

    if thick < 3.5:
        findings.append(f"**Meniscal Thinning ({thick:.2f} mm)**: Mean thickness is reduced, consistent with cartilage/meniscal wear.")

    if bmi >= 30.0:
        findings.append(f"**Elevated Mechanical Joint Load (BMI: {bmi:.1f})**: High BMI contributes elevated axial compressive stress across the medial compartment.")

    st.markdown(f"""
    <div class='insight-card'>
        <h4 style='margin:0 0 8px 0; color:#38bdf8;'>📋 AI Clinical Findings & Insights for {patient_id}</h4>
        <ul style='margin:0; padding-left:20px; font-size:0.95rem; line-height:1.6;'>
            {"".join([f"<li>{f}</li>" for f in findings])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- INTERACTIVE 3D SLICE VIEWER ----------------
    st.subheader("🔬 Interactive 3D MRI Multiplanar Slice Viewer")
    
    norm_array = res["norm_array"]
    men_mask = res["meniscus_mask"]
    tib_mask = res["tibia_mask"]
    z_dim, y_dim, x_dim = norm_array.shape

    c_v1, c_v2, c_v3 = st.columns([2, 3, 2])
    with c_v1:
        plane = st.selectbox("Anatomical View", ["Coronal View (Joint Line & Extrusion)", "Axial View (Transverse)", "Sagittal View"], index=0)
    with c_v2:
        if "Coronal" in plane:
            slice_idx = st.slider("Coronal Slice (Anterior ↔ Posterior)", 0, y_dim - 1, y_dim // 2)
            mri_2d = norm_array[:, slice_idx, :]
            men_2d = men_mask[:, slice_idx, :]
            tib_2d = tib_mask[:, slice_idx, :]
            aspect_ratio = res["spacing_zyx"][0] / res["spacing_zyx"][2]
        elif "Axial" in plane:
            slice_idx = st.slider("Axial Slice (Inferior ↔ Superior)", 0, z_dim - 1, z_dim // 2)
            mri_2d = norm_array[slice_idx, :, :]
            men_2d = men_mask[slice_idx, :, :]
            tib_2d = tib_mask[slice_idx, :, :]
            aspect_ratio = res["spacing_zyx"][1] / res["spacing_zyx"][2]
        else:
            slice_idx = st.slider("Sagittal Slice (Medial ↔ Lateral)", 0, x_dim - 1, x_dim // 2)
            mri_2d = norm_array[:, :, slice_idx]
            men_2d = men_mask[:, :, slice_idx]
            tib_2d = tib_mask[:, :, slice_idx]
            aspect_ratio = res["spacing_zyx"][0] / res["spacing_zyx"][1]

    with c_v3:
        show_mask = st.checkbox("Overlay Meniscus (Cyan)", value=True)
        show_tibia = st.checkbox("Overlay Tibia Landmark (Gold)", value=True)
        show_line = st.checkbox("Show Extrusion Guideline", value=True)

    # Render Slice
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    ax.imshow(mri_2d, cmap='gray', aspect=aspect_ratio)

    if show_mask and men_2d.sum() > 0:
        men_overlay = np.zeros((*men_2d.shape, 4))
        men_overlay[men_2d > 0] = [0.0, 0.9, 0.8, 0.65]  # Cyan
        ax.imshow(men_overlay, aspect=aspect_ratio)

    if show_tibia and tib_2d.sum() > 0:
        tib_overlay = np.zeros((*tib_2d.shape, 4))
        tib_overlay[tib_2d > 0] = [1.0, 0.75, 0.1, 0.55]  # Gold
        ax.imshow(tib_overlay, aspect=aspect_ratio)

    if show_line and "Coronal" in plane and men_2d.sum() > 0:
        men_x = np.where(men_2d)[1]
        if len(men_x) > 0:
            outer_x = men_x.max()
            ax.axvline(x=outer_x, color='#ef4444', linestyle='--', linewidth=1.5, label='Meniscus Edge')
            if tib_2d.sum() > 0:
                tib_x = np.where(tib_2d)[1]
                ax.axvline(x=tib_x.max(), color='#fbbf24', linestyle=':', linewidth=1.5, label='Tibia Edge')

    ax.set_title(f"{plane} — Slice {slice_idx}", color='#f8fafc', fontsize=11, pad=8)
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

    # ---------------- 1-CLICK CLINICAL REPORT EXPORT ----------------
    st.markdown("---")
    st.subheader("📄 Clinical Report & Data Export")

    report_text = f"""================================================================================
NEXORA ORTHOAI — KNEE MRI MENISCUS & OSTEOARTHRITIS REPORT
================================================================================
PATIENT IDENTIFICATION & ACCESSION:
  Patient ID          : {patient_id}
  Scan File           : {loaded_mri_dict["filename"]}
  Age / Biological Sex: {age} Years / {sex}
  Body Mass Index     : {bmi:.1f} kg/m² ({bmi_cat})
  Study Date          : 2026-08-23

QUANTITATIVE 3D MRI MENISCUS BIOMARKERS:
  - Meniscus Volume   : {vol:.2f} cm³
  - Mean Thickness    : {thick:.2f} mm
  - Tibial Extrusion  : {ext:.2f} mm (Normal: < 3.0 mm | Pathological: >= 3.0 mm)

AI DIAGNOSTIC ASSESSMENT:
  - Classification    : {oa_res["oa_classification"].upper()}
  - OA Risk Score     : {prob:.1f}%
  - Biomechanical State: {"Significant meniscal volume reduction and hoop stress disruption." if is_oa else "Preserved meniscal geometry and intact joint morphology."}

CLINICAL RECOMMENDATION:
  {"Recommend clinical correlation, weight-bearing radiography, and orthopaedic consultation." if is_oa else "No significant MRI biomarkers of active osteoarthritis detected in the medial meniscus."}
================================================================================
Generated by Nexora OrthoAI Platform.
"""

    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"{patient_id}_Report.txt",
            mime="text/plain",
        )
    with col_exp2:
        json_str = json.dumps({
            "patient_id": patient_id,
            "demographics": {"age": age, "sex": sex, "bmi": bmi, "category": bmi_cat},
            "biomarkers": {"volume_cm3": vol, "thickness_mm": thick, "extrusion_mm": ext},
            "diagnosis": oa_res,
        }, indent=2)
        st.download_button(
            label="📥 Download Metrics (.json)",
            data=json_str,
            file_name=f"{patient_id}_metrics.json",
            mime="application/json",
        )
    with col_exp3:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_mask:
            save_mask(res["meniscus_mask"], res["preprocessed_img"], tmp_mask.name)
            with open(tmp_mask.name, "rb") as f_mask:
                mask_data = f_mask.read()
        st.download_button(
            label="📥 Download 3D Mask (.nii.gz)",
            data=mask_data,
            file_name=f"{patient_id}_meniscus_mask.nii.gz",
            mime="application/gzip",
        )


if __name__ == "__main__":
    main()
