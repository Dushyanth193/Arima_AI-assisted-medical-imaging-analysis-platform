# 🩺 NEXORA — Unified AI Medical Imaging Analysis Platform

> **AI-Assisted Orthopedic Decision-Support for Knee Osteoarthritis Diagnostics & Total Knee Arthroplasty (TKA) Sizing.**

---

## 🌟 Overview

**NEXORA** integrates two specialized medical imaging pipelines into a unified, single-process Streamlit platform:
1. **Module 1 (MRI Stream)**: 3D Meniscus Segmentation, Tibial Extrusion & Osteoarthritis Risk Classification.
2. **Module 2 (CT Stream)**: 3D MONAI DynUNet Bone Segmentation, Resection Morphometry & Total Knee Implant Matching.

---

## 📁 Final Application Architecture

```
D:\Nexora-hackathon\final_app\
├── app.py                     # Main single-process Streamlit orchestrator & router
├── requirements.txt           # Unified dependency specification
├── README.md                  # System documentation & run guide
├── integration/
│   ├── __init__.py
│   ├── module1_adapter.py     # Clean Python wrapper calling module1_mri functions
│   └── module2_adapter.py     # Clean Python wrapper calling module-2 functions & configuring DB
└── views/
    ├── __init__.py
    ├── dashboard_view.py      # Dual-card overview with quick navigation & pipeline status
    ├── module1_view.py        # Patient form, MRI scan loader, 3D slice viewer, metrics & OA report
    ├── module2_view.py        # CT loader, 3D bone segmentation, measurements, implant catalog matching & ranking
    ├── results_view.py        # Combined session results & comprehensive multi-modal orthopedic report
    └── about_view.py          # Platform overview, architecture, validation benchmarks & clinical disclaimer
```

---

## 🔗 How Modules are Wired In

### Module 1: MRI Analysis & OA Diagnostics
- **Adapter**: `final_app/integration/module1_adapter.py`
- **Underlying Codebase**: `D:\Nexora-hackathon\module1_mri`
- **Key Functions Called**:
  - `src.preprocessing.preprocess_pipeline()`: Isotropic 0.5mm resampling, N4 bias correction, curvature diffusion.
  - `src.segmentation.segment_meniscus_placeholder()`: 3D meniscus segmentation.
  - `src.landmarks.detect_tibial_plateau_landmarks()`: Automated tibial plateau edge boundary extraction.
  - `src.features.extract_features()`: Meniscus volume ($cm^3$), mean thickness ($mm$), and extrusion ($mm$).
  - `src.classification.load_classifier()` & `src.classification.predict_oa()`: Random Forest classification.
- **Models Loaded**: `module1_mri/models/oa_classifier.joblib` & `module1_mri/models/oa_classifier_real.joblib`.

### Module 2: CT Bone Sizing & Implant Matching
- **Adapter**: `final_app/integration/module2_adapter.py`
- **Underlying Codebase**: `D:\Nexora-hackathon\module-2`
- **Key Functions Called**:
  - `src.pipeline.inference_pipeline.run_pipeline_for_file()`: End-to-end CT processing.
  - `src.preprocessing.ct_preprocessing.preprocess_ct()`: Hounsfield Unit windowing & isotropic resampling.
  - `src.segmentation.infer.segment_ct()`: 3D MONAI DynUNet segmentation.
  - `src.measurement.anatomical_measurement.extract_anatomical_measurements()`: Distal Femur and Proximal Tibia ML width, AP dimension, and volumes.
  - `src.matching.implant_matcher.match_patient_to_implants()`: Sizing queries against catalog with tolerance and overhang evaluation.
- **Database Used**: SQLite `module-2/knee_implant.db` with seeded table `implant_components` (Persona, Triathlon, ATTUNE, Journey II).
- **Checkpoints Used**: `module-2/checkpoints/best_model.pt`.

---

## 🚀 How to Run

1. Open your terminal:
```bash
cd D:\Nexora-hackathon\final_app
```

2. Launch Streamlit:
```bash
streamlit run app.py
```

3. Open your browser at:
```
http://localhost:8501
```

---

## ⚠️ Clinical Research Prototype Notice

*This platform is an AI-assisted research and decision-support prototype. It is not intended to provide a standalone medical diagnosis or final implant selection. Results must be reviewed by a qualified medical professional.*
