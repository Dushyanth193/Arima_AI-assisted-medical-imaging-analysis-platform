"""
Streamlit Knee Implant Sizing Application
==========================================
Streamlined interface for Knee Anatomical Measurement & Implant Size Matching.
Inputs: CT Scan file upload (.nii / .nii.gz / .zip).
Outputs:
  - Femoral Mediolateral (ML) Width (mm)
  - Femoral Anteroposterior (AP) Dimension (mm)
  - Tibial Mediolateral (ML) Width (mm)
  - Tibial Anteroposterior (AP) Dimension (mm)
  - Closest Matching Implant Size
"""
from __future__ import annotations

import os
from pathlib import Path
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Knee Implant Sizing",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Knee Implant Sizing & Anatomical Measurement")
st.caption(
    "AI-Assisted Assessment for Femoral/Tibial Sizing — Decision-Support Only."
)

st.divider()

st.subheader("📥 Input: CT Scan Upload")

uploaded_file = st.file_uploader(
    "Upload Knee CT Scan (.nii, .nii.gz, .zip)",
    type=["nii", "gz", "zip"]
)
patient_reference = st.text_input("Patient Reference ID", value="patient_01")

st.divider()

run_button = st.button("⚡ Process CT Scan & Compute Sizing", type="primary", use_container_width=True)

if run_button:
    if uploaded_file is None:
        st.error("Please upload a CT scan file (.nii / .nii.gz / .zip) before running.")
        st.stop()

    with st.spinner("Processing CT Scan..."):
        content_type = "application/zip" if uploaded_file.name.endswith(".zip") else "application/octet-stream"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), content_type)}

        try:
            response = requests.post(
                f"{API_BASE_URL}/patients/{patient_reference}/process",
                files=files,
                timeout=600,
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to backend API at {API_BASE_URL}. Ensure FastAPI is running.")
            st.stop()

        if response.status_code != 200:
            st.error(f"Pipeline processing failed ({response.status_code}): {response.text}")
            st.stop()

        data = response.json()
        meas = data.get("measurements", {})
        fem_rec = data.get("femoral", {}).get("recommended") or {}
        tib_rec = data.get("tibial", {}).get("recommended") or {}

        st.success("✅ CT Processing & Implant Sizing Complete!")
        st.subheader("📊 Output Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Femoral Mediolateral (ML) Width",
                value=f"{meas.get('femur', {}).get('ml_width_mm', 0.0):.1f} mm"
            )
            st.metric(
                label="Femoral Anteroposterior (AP) Dimension",
                value=f"{meas.get('femur', {}).get('ap_dimension_mm', 0.0):.1f} mm"
            )

        with col2:
            st.metric(
                label="Tibial Mediolateral (ML) Width",
                value=f"{meas.get('tibia', {}).get('ml_width_mm', 0.0):.1f} mm"
            )
            st.metric(
                label="Tibial Anteroposterior (AP) Dimension",
                value=f"{meas.get('tibia', {}).get('ap_dimension_mm', 0.0):.1f} mm"
            )

        st.markdown("### 🦴 Closest Matching Implant Size")

        implant_col1, implant_col2 = st.columns(2)
        with implant_col1:
            st.info(
                f"**Femoral Component Size**: **Size {fem_rec.get('size_label', 'N/A')}**\n\n"
                f"- Manufacturer: `{fem_rec.get('manufacturer', 'SampleOrtho')}`\n"
                f"- System: `{fem_rec.get('system_name', 'Generic-PS')}`\n"
                f"- ML Width: `{fem_rec.get('ml_width_mm', 0.0)} mm` | AP: `{fem_rec.get('ap_dimension_mm', 0.0)} mm`"
            )
        with implant_col2:
            st.info(
                f"**Tibial Component Size**: **Size {tib_rec.get('size_label', 'N/A')}**\n\n"
                f"- Manufacturer: `{tib_rec.get('manufacturer', 'SampleOrtho')}`\n"
                f"- System: `{tib_rec.get('system_name', 'Generic-PS')}`\n"
                f"- ML Width: `{tib_rec.get('ml_width_mm', 0.0)} mm` | AP: `{tib_rec.get('ap_dimension_mm', 0.0)} mm`"
            )

