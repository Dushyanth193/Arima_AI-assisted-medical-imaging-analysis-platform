"""
about_view.py
-------------
NEXORA Platform Overview, Multi-Modal Architecture, Mathematical Specifications & Disclaimers.
"""
import streamlit as st


def render_about():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
                border-left: 5px solid #38bdf8; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='margin:0; color:#f8fafc; font-size:1.6rem;'>ℹ️ About NEXORA OrthoAI Platform</h2>
        <p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.92rem;'>
            AI-Assisted Medical Imaging Analysis Platform for Integrated Knee Osteoarthritis Diagnostics and Total Knee Arthroplasty Sizing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 System Architecture & Engineering Design")
    st.markdown("""
    NEXORA unifies two specialized medical imaging pipelines into an end-to-end orthopedic decision-support workflow:
    
    1. **Module 1 (Diagnostic Stream — MRI)**:
       - **Input**: 3D Proton Density / T2 / Fast Spin Echo Knee MRI (`.nii`, `.nii.gz`, `.npz`).
       - **Preprocessing**: Isotropic spatial resampling to $0.5\\text{ mm}$, N4ITK bias field correction, curvature anisotropic diffusion filtering.
       - **3D Segmentation**: Multi-atlas and intensity-thresholded meniscus extraction coupled with automated tibial plateau landmark boundary detection.
       - **Biomarkers**: Meniscus Volume ($V = \\sum v_i \\cdot \\Delta x \\Delta y \\Delta z$), Mean Thickness ($T$), and Meniscal Extrusion ($E = \\max(0, x_{\\text{meniscus}}^{\\text{outer}} - x_{\\text{tibia}}^{\\text{margin}})$).
       - **Classifier**: Random Forest trained on geometric morphometry and demographic risk factors.

    2. **Module 2 (Surgical Planning Stream — CT)**:
       - **Input**: Multi-slice Axial Knee CT DICOM series or 3D NIfTI volume (`.nii.gz`).
       - **Preprocessing**: Hounsfield Unit windowing ($-200$ to $+2000$ HU) and $1.0\\text{ mm}$ isotropic resampling.
       - **Deep Learning**: 3D MONAI DynUNet segmentation architecture generating binary masks for Distal Femur (Label 1) and Proximal Tibia (Label 2).
       - **Morphometry**: Calculates exact Mediolateral (ML) width and Anteroposterior (AP) dimensions in physical millimeter units.
       - **Implant Matching**: Queries SQLite `implant_components` catalog and calculates weighted Euclidean matching distance:
         $$d = \\sqrt{w_{\\text{ML}} (\\text{ML}_{\\text{bone}} - \\text{ML}_{\\text{implant}})^2 + w_{\\text{AP}} (\\text{AP}_{\\text{bone}} - \\text{AP}_{\\text{implant}})^2}$$
         flagging overhang ($>1.5\\text{ mm}$) and under-coverage risk conditions.
    """)

    st.markdown("---")

    st.markdown("### 📚 Supported Implant Catalog")
    st.markdown("""
    The database integrates standardized specification sheets from leading orthopedic manufacturers:
    - **Zimmer Biomet Persona**: Sizes 1–9
    - **Stryker Triathlon**: Sizes 1–8
    - **DePuy Synthes ATTUNE**: Sizes 1–10
    - **Smith & Nephew Journey II**: Sizes 1–8
    """)

    st.markdown("---")

    # Disclaimer
    st.markdown("""
    <div style='background:#18181b; border-left: 4px solid #f59e0b; padding: 16px 20px; border-radius: 8px; margin-top: 20px;'>
        <h4 style='margin:0 0 6px 0; color:#fbbf24;'>⚠️ Clinical & Regulatory Disclaimer</h4>
        <p style='margin:0; color:#a1a1aa; font-size:0.88rem; line-height:1.6;'>
            This platform is an AI-assisted research and decision-support prototype. It is not intended to provide a standalone medical diagnosis or final implant selection. Results must be reviewed by a qualified medical professional.
        </p>
    </div>
    """, unsafe_allow_html=True)
