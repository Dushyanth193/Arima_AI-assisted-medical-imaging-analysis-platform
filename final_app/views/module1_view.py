"""
module1_view.py
---------------
NEXORA Module 1 Streamlit View: 3D MRI Meniscus Segmentation,
Biomarker Extraction (Volume, Thickness, Extrusion) & OA Risk Diagnostics.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from integration.module1_adapter import (
    get_available_models,
    get_module1_demo_scans,
    load_module1_classifier,
    load_mri_scan,
    process_mri_pipeline,
)


def render_module1():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
                border-left: 5px solid #38bdf8; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='margin:0; color:#f8fafc; font-size:1.6rem;'>🧲 Module 1 — MRI Meniscus Biomarkers & OA Diagnostics</h2>
        <p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.92rem;'>
            Automated 3D meniscus segmentation, tibial extrusion calculation, and AI Osteoarthritis risk assessment from knee MRI volumes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    available_models = get_available_models()
    if not available_models:
        st.warning("⚠️ Required model is not available in `module1_mri/models`. Please ensure `oa_classifier.joblib` exists.")

    # Top Controls: 2 Columns (Demographics & Model on Left, MRI Input on Right)
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Patient Profile & AI Model")
        
        c_mod, c_id = st.columns([1.2, 1])
        with c_mod:
            model_options = list(available_models.keys()) if available_models else ["Default Baseline"]
            chosen_model_name = st.selectbox("AI Model Architecture", model_options, index=0)
        with c_id:
            default_pid = st.session_state.get("patient_info", {}).get("patient_id", "PATIENT_MRI_001")
            patient_id = st.text_input("Patient ID / Reference", value=default_pid)

        col_a, col_b = st.columns(2)
        with col_a:
            age = st.slider("Age (Years)", 20, 90, 48)
            sex = st.selectbox("Biological Sex", ["M", "F"], index=0)
        with col_b:
            bmi = st.number_input("BMI (kg/m²)", 15.0, 48.0, 24.5, 0.5)
            bmi_cat = "Normal (<25)" if bmi < 25 else ("Overweight (25-30)" if bmi < 30 else "Obese (≥30)")
            st.caption(f"Category: **{bmi_cat}**")

    with col_right:
        st.subheader("2. 3D MRI Scan Source")
        input_mode = st.radio("Input Method", ["📁 Select Preloaded Patient Scan", "📤 Upload New 3D MRI"], horizontal=True)

        loaded_mri_dict = None
        demo_scans = get_module1_demo_scans()

        if input_mode == "📁 Select Preloaded Patient Scan":
            if demo_scans:
                selected_label = st.selectbox("Choose Demo Scan", list(demo_scans.keys()), index=0)
                selected_path = demo_scans[selected_label]
                try:
                    loaded_mri_dict = load_mri_scan(selected_path, os.path.basename(selected_path))
                    st.success(f"Loaded: `{os.path.basename(selected_path)}`")
                except Exception as e:
                    st.error(f"Failed to load scan: {e}")
            else:
                st.info("No preloaded demo scans found in `module1_mri/data`.")
        else:
            uploaded_file = st.file_uploader("Upload 3D MRI (.nii, .nii.gz, .npz)", type=["nii", "gz", "npz"])
            if uploaded_file is not None:
                try:
                    loaded_mri_dict = load_mri_scan(uploaded_file, uploaded_file.name)
                    st.success(f"Uploaded & Loaded: `{uploaded_file.name}`")
                except Exception as e:
                    st.error(f"Error reading uploaded MRI file: {e}")

    # Metadata Inspection Expander
    if loaded_mri_dict:
        with st.expander("🔍 Inspect 3D MRI Technical Metadata", expanded=False):
            arr = loaded_mri_dict.get("array")
            sitk_img = loaded_mri_dict.get("sitk_image")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.write(f"**Shape (Z, Y, X):** `{arr.shape if arr is not None else 'N/A'}`")
            with m_col2:
                st.write(f"**Voxel Spacing:** `{tuple(round(s, 2) for s in sitk_img.GetSpacing()) if sitk_img else 'N/A'} mm`")
            with m_col3:
                st.write(f"**Data Type:** `{arr.dtype if arr is not None else 'N/A'}`")
            with m_col4:
                st.write(f"**Intensity Range:** `[{arr.min():.1f}, {arr.max():.1f}]`")

    st.markdown("---")

    # Run Pipeline Action
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_btn = st.button("⚡ Run 3D MRI Analysis & OA Diagnostics", type="primary", use_container_width=True)

    if run_btn:
        if not loaded_mri_dict:
            st.error("Please select or upload a 3D MRI scan first.")
            return

        with st.spinner("Executing 3D Resampling, Meniscus Segmentation, Landmark Detection & OA Risk Model..."):
            clf, _ = load_module1_classifier(chosen_model_name)
            result = process_mri_pipeline(
                raw_mri_dict=loaded_mri_dict,
                age=age,
                sex=sex,
                bmi=bmi,
                clf=clf,
            )

        if not result.get("success", False):
            st.error(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")
            return

        # Store in session state for persistence and cross-module report
        st.session_state["mri_results"] = result
        st.session_state["patient_info"] = {"patient_id": patient_id, "age": age, "sex": sex, "bmi": bmi}
        st.success("✅ 3D MRI Segmentation & Quantitative Biomarker Analysis Complete!")

    # Display Results if available in Session State
    if st.session_state.get("mri_results") is not None:
        res = st.session_state["mri_results"]
        oa_res = res["oa_result"]
        feat = res["features"]
        
        vol = feat.get("meniscus_volume_cm3", 0.0)
        thick = feat.get("meniscus_thickness_mm", 0.0)
        ext = feat.get("meniscus_extrusion_mm", 0.0)
        is_oa = "Osteoarthritis" in oa_res.get("oa_classification", "")
        prob = oa_res.get("oa_probability", 0.0) * 100

        st.markdown("### 📊 Diagnostic Output & 3D Biomarkers")

        # Metric Banner Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label="OA Diagnostic Status",
                value=f"{oa_res.get('oa_classification', 'Unknown')}",
                delta=f"Risk Score: {prob:.1f}%",
                delta_color="inverse" if is_oa else "normal",
            )
        with c2:
            st.metric(
                label="Meniscus Volume",
                value=f"{vol:.2f} cm³",
                delta="Preserved Bulk" if vol >= 7.5 else "Volume Loss / Wear",
                delta_color="normal" if vol >= 7.5 else "inverse",
            )
        with c3:
            st.metric(
                label="Mean Thickness",
                value=f"{thick:.2f} mm",
                delta="Normal Cartilage" if thick >= 3.5 else "Thinning / Wear",
                delta_color="normal" if thick >= 3.5 else "inverse",
            )
        with c4:
            st.metric(
                label="Tibial Extrusion",
                value=f"{ext:.2f} mm",
                delta="Normal (<3.0 mm)" if ext < 3.0 else "Pathological (≥3.0 mm)",
                delta_color="normal" if ext < 3.0 else "inverse",
            )

        # AI Clinical Insights Box
        st.markdown("#### 📋 AI Clinical Findings & Insights")
        insights = []
        if is_oa:
            insights.append(f"**High Osteoarthritis Probability ({prob:.1f}%)**: Quantitative imaging patterns indicate joint degeneration.")
        else:
            insights.append(f"**Low Osteoarthritis Probability ({prob:.1f}%)**: Intact joint morphology with healthy meniscal geometry.")

        if ext >= 3.0:
            insights.append(f"**Pathological Extrusion ({ext:.2f} mm)**: Outward meniscal shift (≥ 3.0 mm) reflects circumferential hoop stress compromise.")
        else:
            insights.append(f"**Normal Meniscal Position ({ext:.2f} mm)**: Well-contained within the tibial plateau margin (< 3.0 mm).")

        if thick < 3.5:
            insights.append(f"**Meniscal Thinning ({thick:.2f} mm)**: Sub-threshold mean thickness suggests cartilage wear.")

        st.info("\n\n".join([f"• {item}" for item in insights]))

        # Multiplanar Interactive 3D Viewer
        st.markdown("#### 🔬 Interactive 3D MRI Multiplanar Slice Viewer")
        norm_array = res["norm_array"]
        men_mask = res["meniscus_mask"]
        tibia_mask = res["tibia_mask"]
        z_dim, y_dim, x_dim = norm_array.shape

        v_col1, v_col2, v_col3 = st.columns([2, 3, 2])
        with v_col1:
            plane = st.selectbox("Anatomical Plane", ["Coronal View (Joint Line & Extrusion)", "Axial View (Transverse)", "Sagittal View (Medial/Lateral)"])
        with v_col2:
            if "Coronal" in plane:
                slice_idx = st.slider("Coronal Slice", 0, y_dim - 1, y_dim // 2)
                mri_2d = norm_array[:, slice_idx, :]
                men_2d = men_mask[:, slice_idx, :]
                tib_2d = tibia_mask[:, slice_idx, :] if tibia_mask is not None else np.zeros_like(men_2d)
                aspect = res["spacing_zyx"][0] / res["spacing_zyx"][2]
            elif "Axial" in plane:
                slice_idx = st.slider("Axial Slice", 0, z_dim - 1, z_dim // 2)
                mri_2d = norm_array[slice_idx, :, :]
                men_2d = men_mask[slice_idx, :, :]
                tib_2d = tibia_mask[slice_idx, :, :] if tibia_mask is not None else np.zeros_like(men_2d)
                aspect = res["spacing_zyx"][1] / res["spacing_zyx"][2]
            else:
                slice_idx = st.slider("Sagittal Slice", 0, x_dim - 1, x_dim // 2)
                mri_2d = norm_array[:, :, slice_idx]
                men_2d = men_mask[:, :, slice_idx]
                tib_2d = tibia_mask[:, :, slice_idx] if tibia_mask is not None else np.zeros_like(men_2d)
                aspect = res["spacing_zyx"][0] / res["spacing_zyx"][1]

        with v_col3:
            show_men = st.checkbox("Overlay Meniscus (Cyan)", value=True)
            show_tib = st.checkbox("Overlay Tibia (Gold)", value=True)
            show_line = st.checkbox("Show Extrusion Guideline", value=True)

        fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor='#0b1120')
        ax.set_facecolor('#0b1120')
        ax.imshow(mri_2d, cmap='gray', aspect=aspect)

        if show_men and men_2d.sum() > 0:
            men_overlay = np.zeros((*men_2d.shape, 4))
            men_overlay[men_2d > 0] = [0.0, 0.9, 0.8, 0.65]  # Cyan
            ax.imshow(men_overlay, aspect=aspect)

        if show_tib and tib_2d.sum() > 0:
            tib_overlay = np.zeros((*tib_2d.shape, 4))
            tib_overlay[tib_2d > 0] = [1.0, 0.75, 0.1, 0.55]  # Gold
            ax.imshow(tib_overlay, aspect=aspect)

        if show_line and "Coronal" in plane and men_2d.sum() > 0:
            men_x = np.where(men_2d)[1]
            if len(men_x) > 0:
                ax.axvline(x=men_x.max(), color='#ef4444', linestyle='--', linewidth=1.5, label='Meniscus Edge')
                if tib_2d.sum() > 0:
                    tib_x = np.where(tib_2d)[1]
                    ax.axvline(x=tib_x.max(), color='#fbbf24', linestyle=':', linewidth=1.5, label='Tibia Edge')

        ax.set_title(f"{plane} — Slice {slice_idx}", color='#f8fafc', fontsize=11, pad=8)
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)
