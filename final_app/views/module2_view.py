"""
module2_view.py
---------------
NEXORA Module 2 Streamlit View: 3D CT Bone Segmentation,
Femoral/Tibial Morphometry & Knee Implant Size Matching.
"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from integration.module2_adapter import (
    get_catalog_summary,
    get_checkpoint_status,
    get_module2_demo_patients,
    process_ct_pipeline,
)


def render_module2():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #064e3b 0%, #0f172a 100%);
                border-left: 5px solid #10b981; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='margin:0; color:#f8fafc; font-size:1.6rem;'>🦴 Module 2 — CT Bone Sizing & Implant Matching</h2>
        <p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.92rem;'>
            Deep learning 3D bone segmentation (MONAI DynUNet), femoral/tibial anatomical dimension extraction, and total knee implant catalog matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

    catalog_info = get_catalog_summary()
    ckpt_info = get_checkpoint_status()

    # Controls: 2 Columns
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Patient & Catalog Sizing Filter")
        default_pid = st.session_state.get("patient_info", {}).get("patient_id", "PATIENT_CT_001")
        patient_ref = st.text_input("Patient Reference ID", value=default_pid)

        col_mfg, col_sys = st.columns(2)
        with col_mfg:
            mfg_options = ["All"] + (catalog_info.get("manufacturers", []) if catalog_info.get("available") else [])
            selected_mfg = st.selectbox("Manufacturer Filter", mfg_options, index=0)
        with col_sys:
            sys_options = ["All"] + (catalog_info.get("systems", []) if catalog_info.get("available") else [])
            selected_sys = st.selectbox("Implant System Filter", sys_options, index=0)

        # Catalog Stats Pill
        if catalog_info.get("available"):
            st.caption(f"📚 Connected Catalog: **{catalog_info.get('total_components', 0)} Components** across {len(catalog_info.get('manufacturers', []))} Vendors")
        else:
            st.warning("⚠️ Implant database not detected. Check SQLite connection.")

    with col_right:
        st.subheader("2. Knee CT Scan Input")
        input_mode = st.radio("CT Source Method", ["📁 Select Preloaded Patient DICOM", "📤 Upload CT Scan (.nii / .zip)"], horizontal=True)

        demo_patients = get_module2_demo_patients()
        selected_input = None
        input_name = ""

        if input_mode == "📁 Select Preloaded Patient DICOM":
            if demo_patients:
                selected_label = st.selectbox("Choose Patient Case", list(demo_patients.keys()), index=0)
                selected_input = demo_patients[selected_label]
                input_name = selected_input.name
                st.success(f"Selected: `{selected_input.name}` ({len(list(selected_input.glob('*.DCM*')))} DICOM slices)")
            else:
                st.info("No preloaded DICOM cases found in `module-2/Patient_Dataset`.")
        else:
            uploaded_file = st.file_uploader("Upload CT Scan (.nii, .nii.gz, .zip of DICOMs)", type=["nii", "gz", "zip"])
            if uploaded_file is not None:
                selected_input = uploaded_file
                input_name = uploaded_file.name
                st.success(f"Uploaded: `{uploaded_file.name}`")

    st.markdown("---")

    # Run Pipeline Action
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_btn = st.button("⚡ Process CT Scan & Compute Implant Sizing", type="primary", use_container_width=True)

    if run_btn:
        if selected_input is None:
            st.error("Please select or upload a CT scan before running.")
            return

        with st.spinner("Executing 3D MONAI DynUNet Segmentation, Bone Resection Morphometry & Catalog Matching..."):
            result = process_ct_pipeline(
                input_source=selected_input,
                filename_or_ref=input_name,
                patient_ref=patient_ref,
                manufacturer_filter=selected_mfg,
                system_filter=selected_sys,
            )

        if not result.get("success", False):
            st.error(f"Module 2 execution error: {result.get('error', 'Unknown error')}")
            return

        st.session_state["ct_results"] = result
        st.success("✅ 3D Bone Segmentation, Resection Morphometry & Implant Sizing Complete!")

    # Display Results if in Session State
    if st.session_state.get("ct_results") is not None:
        res = st.session_state["ct_results"]
        meas = res.get("measurements", {})
        fem_rec = res.get("femoral", {}).get("recommended") or {}
        tib_rec = res.get("tibial", {}).get("recommended") or {}
        fem_alts = res.get("femoral", {}).get("alternatives", [])
        tib_alts = res.get("tibial", {}).get("alternatives", [])

        st.markdown("### 📊 Extracted Anatomical Resection Measurements")

        # Metric Cards
        col_f, col_t = st.columns(2)
        with col_f:
            st.markdown("""
            <div style='background:#1e293b; padding:14px; border-radius:8px; border-top:3px solid #38bdf8; margin-bottom:12px;'>
                <h4 style='margin:0 0 10px 0; color:#f8fafc;'>🦴 Distal Femoral Dimensions</h4>
            </div>
            """, unsafe_allow_html=True)
            f_m1, f_m2, f_m3 = st.columns(3)
            with f_m1:
                st.metric("ML Width", f"{meas.get('femur_ml_width_mm', 0.0):.1f} mm")
            with f_m2:
                st.metric("AP Dimension", f"{meas.get('femur_ap_dimension_mm', 0.0):.1f} mm")
            with f_m3:
                st.metric("Bone Volume", f"{meas.get('femur_volume_mm3', 0.0) / 1000.0:.1f} cm³")

        with col_t:
            st.markdown("""
            <div style='background:#1e293b; padding:14px; border-radius:8px; border-top:3px solid #10b981; margin-bottom:12px;'>
                <h4 style='margin:0 0 10px 0; color:#f8fafc;'>🦴 Proximal Tibial Dimensions</h4>
            </div>
            """, unsafe_allow_html=True)
            t_m1, t_m2, t_m3 = st.columns(3)
            with t_m1:
                st.metric("ML Width", f"{meas.get('tibia_ml_width_mm', 0.0):.1f} mm")
            with t_m2:
                st.metric("AP Dimension", f"{meas.get('tibia_ap_dimension_mm', 0.0):.1f} mm")
            with t_m3:
                st.metric("Bone Volume", f"{meas.get('tibia_volume_mm3', 0.0) / 1000.0:.1f} cm³")

        st.markdown("### 🏆 Optimal Implant Size Recommendations")

        # Top Match Cards
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"""
            <div style='background:#111827; border: 1px solid #38bdf8; border-radius:10px; padding:18px; box-shadow:0 4px 12px rgba(0,0,0,0.3);'>
                <div style='font-size:0.8rem; color:#38bdf8; font-weight:700; text-transform:uppercase;'>Recommended Femoral Component</div>
                <h2 style='margin:4px 0 8px 0; color:#f8fafc;'>Size {fem_rec.get('size_label', 'N/A')}</h2>
                <div style='font-size:0.9rem; color:#cbd5e1; line-height:1.6;'>
                    • <strong>Manufacturer:</strong> {fem_rec.get('manufacturer', 'N/A')}<br>
                    • <strong>System:</strong> {fem_rec.get('system_name', 'N/A')}<br>
                    • <strong>Implant ML / AP:</strong> {fem_rec.get('ml_width_mm', 0.0):.1f} mm / {fem_rec.get('ap_dimension_mm', 0.0):.1f} mm<br>
                    • <strong>Matching Distance Score:</strong> <code>{fem_rec.get('matching_score', 0.0):.2f} mm</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div style='background:#111827; border: 1px solid #10b981; border-radius:10px; padding:18px; box-shadow:0 4px 12px rgba(0,0,0,0.3);'>
                <div style='font-size:0.8rem; color:#10b981; font-weight:700; text-transform:uppercase;'>Recommended Tibial Baseplate</div>
                <h2 style='margin:4px 0 8px 0; color:#f8fafc;'>Size {tib_rec.get('size_label', 'N/A')}</h2>
                <div style='font-size:0.9rem; color:#cbd5e1; line-height:1.6;'>
                    • <strong>Manufacturer:</strong> {tib_rec.get('manufacturer', 'N/A')}<br>
                    • <strong>System:</strong> {tib_rec.get('system_name', 'N/A')}<br>
                    • <strong>Implant ML / AP:</strong> {tib_rec.get('ml_width_mm', 0.0):.1f} mm / {tib_rec.get('ap_dimension_mm', 0.0):.1f} mm<br>
                    • <strong>Matching Distance Score:</strong> <code>{tib_rec.get('matching_score', 0.0):.2f} mm</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Ranked Candidates Table
        st.markdown("#### 📋 Ranked Candidate Components & Fit Tolerances")
        tab_fem, tab_tib = st.tabs(["Femoral Candidates", "Tibial Candidates"])

        with tab_fem:
            all_fem = ([fem_rec] if fem_rec else []) + fem_alts
            if all_fem:
                df_fem = pd.DataFrame(all_fem)
                display_cols = [c for c in ["size_label", "manufacturer", "system_name", "ml_width_mm", "ap_dimension_mm", "matching_score", "within_tolerance", "overhang_risk"] if c in df_fem.columns]
                st.dataframe(df_fem[display_cols], use_container_width=True)
            else:
                st.info("No femoral candidates found matching the selected filters.")

        with tab_tib:
            all_tib = ([tib_rec] if tib_rec else []) + tib_alts
            if all_tib:
                df_tib = pd.DataFrame(all_tib)
                display_cols = [c for c in ["size_label", "manufacturer", "system_name", "ml_width_mm", "ap_dimension_mm", "matching_score", "within_tolerance", "overhang_risk"] if c in df_tib.columns]
                st.dataframe(df_tib[display_cols], use_container_width=True)
            else:
                st.info("No tibial candidates found matching the selected filters.")
