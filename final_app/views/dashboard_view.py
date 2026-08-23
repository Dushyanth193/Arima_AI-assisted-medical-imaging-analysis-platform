"""
dashboard_view.py
-----------------
NEXORA Unified Dashboard view with interactive module launch cards,
pipeline status indicators, and pre-surgical workflow overview.
"""
import streamlit as st
from integration.module1_adapter import get_available_models as get_m1_models
from integration.module2_adapter import get_catalog_summary as get_m2_catalog, get_checkpoint_status as get_m2_ckpt


def render_dashboard():
    # Hero Title Banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2b48 100%);
                padding: 24px 28px; border-radius: 12px; border: 1px solid #1e3a5f; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);'>
        <div style='display:flex; align-items:center; justify-content:space-between;'>
            <div>
                <h1 style='margin:0; font-size:2.2rem; color:#f8fafc; font-weight:800; letter-spacing:-0.5px;'>
                    NEXORA <span style='font-weight:400; font-size:1.4rem; color:#38bdf8;'>| OrthoAI</span>
                </h1>
                <p style='margin:6px 0 0 0; color:#94a3b8; font-size:1.05rem;'>
                    Unified AI-Assisted Medical Imaging Platform for Knee Osteoarthritis Diagnostics & Total Knee Arthroplasty (TKA) Implant Sizing
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Indicators
    m1_models = get_m1_models()
    m2_cat = get_m2_catalog()
    m2_ckpt = get_m2_ckpt()

    m1_ran = st.session_state.get("mri_results") is not None
    m2_ran = st.session_state.get("ct_results") is not None

    # Pipeline Workflow Banner
    st.markdown("### 🔀 Integrated Pre-Surgical Orthopedic Workflow")
    st.caption("Select a specialized imaging module below to initiate diagnostic analysis or surgical resection sizing.")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(f"""
        <div style='background: #111827; border: 1px solid {"#22c55e" if m1_ran else "#1f2937"};
                    border-radius: 10px; padding: 22px; height: 100%; display:flex; flex-direction:column; justify-content:space-between;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <span style='background:#0369a1; color:#e0f2fe; padding:4px 10px; border-radius:6px; font-size:0.75rem; font-weight:700; text-transform:uppercase;'>
                        Module 1 • Diagnostic
                    </span>
                    <span style='color:{"#4ade80" if m1_ran else "#94a3b8"}; font-size:0.85rem; font-weight:600;'>
                        {"● Results Available" if m1_ran else "○ Ready for Input"}
                    </span>
                </div>
                <h3 style='margin:0 0 10px 0; color:#f8fafc;'>🧲 MRI Meniscus & OA Diagnostics</h3>
                <p style='color:#94a3b8; font-size:0.92rem; line-height:1.55; margin-bottom:16px;'>
                    Automated 3D meniscus segmentation, tibial plateau landmark boundary extraction, and quantitative morphological biomarker calculation (Volume, Thickness, Extrusion) with AI Osteoarthritis risk classification.
                </p>
                <div style='background:#1e293b; padding:10px 14px; border-radius:6px; margin-bottom:18px;'>
                    <div style='font-size:0.82rem; color:#cbd5e1;'><strong>Input:</strong> 3D Knee MRI (<code>.nii</code>, <code>.nii.gz</code>, <code>.npz</code>)</div>
                    <div style='font-size:0.82rem; color:#cbd5e1; margin-top:4px;'><strong>Output:</strong> 3D Meniscus Mask, Extrusion (mm), OA Risk Score (%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Open Module 1 (MRI Analysis)", key="btn_open_m1", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "Module 1"
            st.rerun()

    with col2:
        st.markdown(f"""
        <div style='background: #111827; border: 1px solid {"#22c55e" if m2_ran else "#1f2937"};
                    border-radius: 10px; padding: 22px; height: 100%; display:flex; flex-direction:column; justify-content:space-between;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <span style='background:#047857; color:#d1fae5; padding:4px 10px; border-radius:6px; font-size:0.75rem; font-weight:700; text-transform:uppercase;'>
                        Module 2 • Surgical
                    </span>
                    <span style='color:{"#4ade80" if m2_ran else "#94a3b8"}; font-size:0.85rem; font-weight:600;'>
                        {"● Results Available" if m2_ran else "○ Ready for Input"}
                    </span>
                </div>
                <h3 style='margin:0 0 10px 0; color:#f8fafc;'>🦴 CT Bone Sizing & Implant Matching</h3>
                <p style='color:#94a3b8; font-size:0.92rem; line-height:1.55; margin-bottom:16px;'>
                    Deep learning (MONAI DynUNet) 3D bone segmentation for Distal Femur and Proximal Tibia. Extracts Mediolateral (ML) and Anteroposterior (AP) dimensions to automatically query and rank catalog implant components.
                </p>
                <div style='background:#1e293b; padding:10px 14px; border-radius:6px; margin-bottom:18px;'>
                    <div style='font-size:0.82rem; color:#cbd5e1;'><strong>Input:</strong> Knee CT Scan (DICOM Series / <code>.nii.gz</code> / <code>.zip</code>)</div>
                    <div style='font-size:0.82rem; color:#cbd5e1; margin-top:4px;'><strong>Output:</strong> AP/ML Dimensions, Catalog Implant Sizes & Ranked Fit</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Open Module 2 (CT Sizing)", key="btn_open_m2", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "Module 2"
            st.rerun()

    st.markdown("---")

    # Current Session Status & Quick Summary
    st.subheader("📊 Current Active Session State")
    s_col1, s_col2, s_col3 = st.columns(3)
    
    with s_col1:
        st.metric(
            label="Module 1 Status",
            value="Analyzed" if m1_ran else "Not Executed",
            delta=f"{st.session_state['mri_results']['oa_result']['oa_classification']}" if m1_ran else None,
        )
    with s_col2:
        st.metric(
            label="Module 2 Status",
            value="Sized" if m2_ran else "Not Executed",
            delta=f"Femur: Size {st.session_state['ct_results'].get('femoral', {}).get('recommended', {}).get('size_label', 'N/A')}" if m2_ran else None,
        )
    with s_col3:
        has_both = m1_ran and m2_ran
        st.metric(
            label="Multi-Modal Report",
            value="Ready" if has_both else ("Partial" if (m1_ran or m2_ran) else "Pending"),
            delta="Complete Pre-Op Profile" if has_both else None,
        )

    # Clinical Research Prototype Disclaimer
    st.markdown("""
    <div style='background:#18181b; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 6px; margin-top: 25px;'>
        <div style='font-size:0.88rem; color:#fbbf24; font-weight:700; margin-bottom:4px;'>⚠️ Clinical & Research Prototype Notice</div>
        <div style='font-size:0.82rem; color:#a1a1aa; line-height:1.5;'>
            This platform is an AI-assisted research and decision-support prototype. It is not intended to provide a standalone medical diagnosis or final implant selection. Results must be reviewed by a qualified medical professional.
        </div>
    </div>
    """, unsafe_allow_html=True)
