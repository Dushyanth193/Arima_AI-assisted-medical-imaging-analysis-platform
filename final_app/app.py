"""
NEXORA — Unified AI-Assisted Medical Imaging Analysis Platform
==============================================================
Single-process Streamlit application orchestrating:
  - Module 1: 3D Knee MRI Meniscus Segmentation & Osteoarthritis Diagnostics
  - Module 2: 3D Knee CT Bone Segmentation, Resection Sizing & TKA Implant Matching
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root and local packages to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from views.dashboard_view import render_dashboard
from views.module1_view import render_module1
from views.module2_view import render_module2
from views.results_view import render_results
from views.about_view import render_about


def init_session_state():
    """Initializes cross-module session storage."""
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Dashboard"
    if "mri_results" not in st.session_state:
        st.session_state["mri_results"] = None
    if "ct_results" not in st.session_state:
        st.session_state["ct_results"] = None
    if "patient_info" not in st.session_state:
        st.session_state["patient_info"] = {
            "patient_id": "PATIENT_001",
            "age": 52,
            "sex": "M",
            "bmi": 26.5,
        }


def main():
    st.set_page_config(
        page_title="NEXORA | AI Medical Imaging Platform",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Global Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1e293b;
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            color: #94a3b8;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0284c7 !important;
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Brand & Navigation
    st.sidebar.markdown("""
    <div style='padding: 10px 0 16px 0; border-bottom: 1px solid #1e293b; margin-bottom: 16px;'>
        <h2 style='margin:0; font-size:1.5rem; color:#f8fafc; font-weight:800; letter-spacing:-0.5px;'>
            🩺 NEXORA
        </h2>
        <div style='color:#38bdf8; font-size:0.82rem; font-weight:600; margin-top:2px;'>
            AI-Assisted Medical Imaging Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = ["Dashboard", "Module 1", "Module 2", "Results", "About"]
    icons = ["🏠", "🧲", "🦴", "📄", "ℹ️"]
    nav_labels = {
        "Dashboard": "🏠  Dashboard",
        "Module 1": "🧲  Module 1 (MRI • OA)",
        "Module 2": "🦴  Module 2 (CT • Sizing)",
        "Results": "📄  Results & Report",
        "About": "ℹ️  About Platform",
    }

    current_idx = nav_items.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_items else 0

    selected_nav = st.sidebar.radio(
        "Navigation",
        options=nav_items,
        format_func=lambda x: nav_labels[x],
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state["nav_selection"] = selected_nav

    st.sidebar.markdown("---")

    # Session State Indicator in Sidebar
    st.sidebar.subheader("Session Status")
    m1_done = st.session_state["mri_results"] is not None
    m2_done = st.session_state["ct_results"] is not None

    st.sidebar.markdown(f"• **Module 1 (MRI):** {'✅ Ready' if m1_done else '⚪ Pending'}")
    st.sidebar.markdown(f"• **Module 2 (CT):** {'✅ Ready' if m2_done else '⚪ Pending'}")

    if m1_done or m2_done:
        if st.sidebar.button("🔄 Reset Active Session", use_container_width=True):
            st.session_state["mri_results"] = None
            st.session_state["ct_results"] = None
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("NEXORA v2.0 • OrthoAI Multi-Modal")

    # Route to Views
    if selected_nav == "Dashboard":
        render_dashboard()
    elif selected_nav == "Module 1":
        render_module1()
    elif selected_nav == "Module 2":
        render_module2()
    elif selected_nav == "Results":
        render_results()
    elif selected_nav == "About":
        render_about()

    # Footer Disclaimer
    st.markdown("""
    <div style='margin-top:40px; padding:12px 16px; border-top:1px solid #1e293b; text-align:center;'>
        <span style='font-size:0.78rem; color:#64748b;'>
            <strong>Disclaimer:</strong> This platform is an AI-assisted research and decision-support prototype. It is not intended to provide a standalone medical diagnosis or final implant selection. Results must be reviewed by a qualified medical professional.
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
