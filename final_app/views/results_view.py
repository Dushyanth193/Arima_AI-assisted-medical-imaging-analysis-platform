"""
results_view.py
---------------
NEXORA Integrated Clinical Results View: Combines MRI Osteoarthritis Diagnostics
and CT Total Knee Implant Sizing into a unified patient summary.
"""
import json
import streamlit as st


def render_results():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%);
                border-left: 5px solid #818cf8; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='margin:0; color:#f8fafc; font-size:1.6rem;'>📄 Integrated Patient Results & Multi-Modal Report</h2>
        <p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.92rem;'>
            Synthesized orthopedic pre-surgical diagnostic report combining MRI soft-tissue biomarkers and CT bony resection sizing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1_res = st.session_state.get("mri_results")
    m2_res = st.session_state.get("ct_results")
    pat_info = st.session_state.get("patient_info", {})

    if m1_res is None and m2_res is None:
        st.info("ℹ️ **No pipeline results available in the current session.** Please run **Module 1 (MRI Analysis)** or **Module 2 (CT Sizing)** from the sidebar navigation to generate clinical reports.")
        return

    # Multi-Modal Summary Banner
    if m1_res is not None and m2_res is not None:
        st.markdown("""
        <div style='background:#064e3b; border: 1px solid #10b981; border-radius:8px; padding:12px 18px; margin-bottom:20px;'>
            <span style='font-size:1rem; font-weight:700; color:#d1fae5;'>🌟 Full Multi-Modal Pre-Surgical Dataset Available</span>
            <div style='font-size:0.85rem; color:#a7f3d0;'>Both 3D MRI Meniscus Diagnostics and 3D CT Bone Sizing have been successfully processed for this patient case.</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Patient Profile Summary
    st.subheader("1. Patient Demographic Profile")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.write(f"**Patient ID:** `{pat_info.get('patient_id', 'PATIENT_001')}`")
    with p_col2:
        st.write(f"**Age:** `{pat_info.get('age', 'N/A')} Years`")
    with p_col3:
        st.write(f"**Biological Sex:** `{pat_info.get('sex', 'N/A')}`")
    with p_col4:
        st.write(f"**BMI:** `{pat_info.get('bmi', 'N/A')} kg/m²`")

    st.markdown("---")

    # 2. Module 1 Diagnostic Section (if run)
    if m1_res is not None:
        st.subheader("2. MRI Meniscus Degradation & Osteoarthritis Status")
        feat = m1_res.get("features", {})
        oa_res = m1_res.get("oa_result", {})
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("OA Diagnostic Status", oa_res.get("oa_classification", "N/A"), f"Risk: {oa_res.get('oa_probability', 0.0)*100:.1f}%")
        with m_c2:
            st.metric("Meniscus Volume", f"{feat.get('meniscus_volume_cm3', 0.0):.2f} cm³")
        with m_c3:
            st.metric("Mean Thickness", f"{feat.get('meniscus_thickness_mm', 0.0):.2f} mm")
        with m_c4:
            st.metric("Tibial Extrusion", f"{feat.get('meniscus_extrusion_mm', 0.0):.2f} mm")

    # 3. Module 2 Sizing Section (if run)
    if m2_res is not None:
        st.subheader("3. CT 3D Bone Resection Morphometry & Implant Sizing")
        meas = m2_res.get("measurements", {})
        fem_rec = m2_res.get("femoral", {}).get("recommended") or {}
        tib_rec = m2_res.get("tibial", {}).get("recommended") or {}

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Femoral ML Width", f"{meas.get('femur_ml_width_mm', 0.0):.1f} mm")
        with c_m2:
            st.metric("Femoral AP Dim", f"{meas.get('femur_ap_dimension_mm', 0.0):.1f} mm")
        with c_m3:
            st.metric("Tibial ML Width", f"{meas.get('tibia_ml_width_mm', 0.0):.1f} mm")
        with c_m4:
            st.metric("Tibial AP Dim", f"{meas.get('tibia_ap_dimension_mm', 0.0):.1f} mm")

        # Top Implant Match Banner
        st.markdown("#### 🏆 Sized Total Knee Arthroplasty (TKA) Components")
        i_col1, i_col2 = st.columns(2)
        with i_col1:
            st.info(
                f"**Recommended Femoral Component**: **Size {fem_rec.get('size_label', 'N/A')}**\n\n"
                f"- Manufacturer: `{fem_rec.get('manufacturer', 'N/A')}` | System: `{fem_rec.get('system_name', 'N/A')}`\n"
                f"- Component ML / AP: `{fem_rec.get('ml_width_mm', 0.0):.1f} mm` / `{fem_rec.get('ap_dimension_mm', 0.0):.1f} mm`\n"
                f"- Geometric Fit Score: `{fem_rec.get('matching_score', 0.0):.2f} mm`"
            )
        with i_col2:
            st.info(
                f"**Recommended Tibial Baseplate**: **Size {tib_rec.get('size_label', 'N/A')}**\n\n"
                f"- Manufacturer: `{tib_rec.get('manufacturer', 'N/A')}` | System: `{tib_rec.get('system_name', 'N/A')}`\n"
                f"- Component ML / AP: `{tib_rec.get('ml_width_mm', 0.0):.1f} mm` / `{tib_rec.get('ap_dimension_mm', 0.0):.1f} mm`\n"
                f"- Geometric Fit Score: `{tib_rec.get('matching_score', 0.0):.2f} mm`"
            )

    st.markdown("---")

    # 4. Master Report Generation & Download
    st.subheader("📄 1-Click Master Clinical Case Report")

    report_lines = [
        "=" * 80,
        "NEXORA ORTHOAI — INTEGRATED PRE-SURGICAL MEDICAL IMAGING REPORT",
        "=" * 80,
        f"PATIENT IDENTIFIER : {pat_info.get('patient_id', 'PATIENT_001')}",
        f"DEMOGRAPHICS       : Age {pat_info.get('age', 'N/A')} | Sex {pat_info.get('sex', 'N/A')} | BMI {pat_info.get('bmi', 'N/A')} kg/m²",
        "-" * 80,
    ]

    if m1_res is not None:
        feat = m1_res.get("features", {})
        oa_res = m1_res.get("oa_result", {})
        report_lines.extend([
            "MODULE 1: MRI MENISCUS & OSTEOARTHRITIS ASSESSMENT",
            f"  - OA Diagnostic Classification : {oa_res.get('oa_classification', 'N/A')}",
            f"  - Estimated OA Risk Probability : {oa_res.get('oa_probability', 0.0)*100:.1f}%",
            f"  - Meniscus Volume              : {feat.get('meniscus_volume_cm3', 0.0):.2f} cm³",
            f"  - Mean Meniscus Thickness      : {feat.get('meniscus_thickness_mm', 0.0):.2f} mm",
            f"  - Meniscal Extrusion           : {feat.get('meniscus_extrusion_mm', 0.0):.2f} mm",
            "-" * 80,
        ])

    if m2_res is not None:
        meas = m2_res.get("measurements", {})
        fem_rec = m2_res.get("femoral", {}).get("recommended") or {}
        tib_rec = m2_res.get("tibial", {}).get("recommended") or {}
        report_lines.extend([
            "MODULE 2: CT BONE MORPHOMETRY & TKA IMPLANT SIZING",
            f"  - Distal Femoral ML Width      : {meas.get('femur_ml_width_mm', 0.0):.1f} mm",
            f"  - Distal Femoral AP Dimension  : {meas.get('femur_ap_dimension_mm', 0.0):.1f} mm",
            f"  - Proximal Tibial ML Width     : {meas.get('tibia_ml_width_mm', 0.0):.1f} mm",
            f"  - Proximal Tibial AP Dimension : {meas.get('tibia_ap_dimension_mm', 0.0):.1f} mm",
            f"  - Recommended Femoral Component: Size {fem_rec.get('size_label', 'N/A')} ({fem_rec.get('manufacturer', 'N/A')} {fem_rec.get('system_name', 'N/A')})",
            f"  - Recommended Tibial Baseplate : Size {tib_rec.get('size_label', 'N/A')} ({tib_rec.get('manufacturer', 'N/A')} {tib_rec.get('system_name', 'N/A')})",
            "-" * 80,
        ])

    report_lines.extend([
        "CLINICAL DECISION-SUPPORT NOTICE:",
        "  This platform is an AI-assisted research and decision-support prototype.",
        "  It is not intended to provide a standalone medical diagnosis or final",
        "  implant selection. Results must be reviewed by a qualified medical professional.",
        "=" * 80,
    ])

    full_report_str = "\n".join(report_lines)

    st.code(full_report_str, language="text")

    st.download_button(
        label="📥 Download Clinical Summary Report (.txt)",
        data=full_report_str,
        file_name=f"Nexora_Clinical_Report_{pat_info.get('patient_id', 'case')}.txt",
        mime="text/plain",
        type="primary",
    )
