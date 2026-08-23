# Knee Implant Sizing — Problem B Prototype

AI-assisted femoral/tibial CT measurement and implant size matching, decision-support only.
**Not a medical device. Final implant selection remains with a qualified orthopedic surgeon.**

---

## 1. Project Overview

This repository implements **Module 2 (Problem B)** from the AI-Assisted Assessment of Medial
Meniscus Thickness and Patient-Specific Knee Implant Sizing problem statement: automated
femur/tibia segmentation from knee CT, anatomical measurement extraction, and matching against
a structured implant sizing database — followed by clinician verification.

## 2. Problem Definition

**Technical statement:** Given a knee CT volume, segment the femur and tibia, extract
mediolateral (ML) width and anteroposterior (AP) dimension for each bone, and rank candidate
implant component sizes from a structured catalog by geometric fit, surfacing the result to a
surgeon for pre-operative TKA planning review — without making an autonomous sizing decision.

## 3. Existing Flow Diagram Interpretation

The provided diagram has two parallel branches feeding one shared downstream pipeline:

- **Reference/training branch** (left column): builds the historical dataset used to *train*
  the segmentation model and to *populate* the implant sizing reference database. This is a
  one-time (or periodically refreshed) offline process, not something that runs per-request.
- **New-patient branch** (right column): the online inference path — a new CT comes in, gets
  preprocessed, segmented, measured, and compared against the same reference/implant data.
- Both branches converge at **Implant Size Matching Model**, which is the only stage that needs
  both the new patient's measurements *and* the reference implant database at request time.

This structure is preserved exactly in the code: `src/segmentation/train.py` +
`src/database/seed_implant_catalog.py` implement the left branch (offline), and
`src/api/main.py`'s `/patients/{id}/process` endpoint implements the right branch (online),
calling into `src/segmentation/infer.py` → `src/measurement/` → `src/matching/` in that order.

**One diagram box needed clarification, not redesign:** "Anatomical Feature Extraction" and
"New-Patient Measurement Extraction" are drawn as AI-adjacent boxes in the same visual style as
segmentation. They are implemented as **deterministic geometry** (scikit-image/SciPy/NumPy on
the segmentation mask), not a second trained model — this matches the specified tech stack row
("Anatomical measurement: SciPy + NumPy") and avoids inventing an unneeded second network.

## 4. Technology Stack

| Stage | Technology used | Notes / deviations |
|---|---|---|
| Medical image input | DICOM / NIfTI | Both supported by `ct_preprocessing.load_ct_volume` |
| Image loading | SimpleITK | As specified |
| Preprocessing | SimpleITK + NumPy | As specified |
| AI segmentation | PyTorch + MONAI (**DynUNet**) | See flagged issue below |
| Mask processing | NumPy + scikit-image | As specified |
| Anatomical measurement | SciPy + NumPy | As specified |
| 3D reconstruction | PyVista (wraps VTK) | As specified |
| Measurement visualization | PyVista (+ Plotly optional in Streamlit) | As specified |
| Implant database | PostgreSQL (via SQLAlchemy) | As specified |
| Implant matching | NumPy + SciPy | As specified |
| Ranking/optimization | SciPy (`scipy.spatial.distance`) | As specified |
| Backend | FastAPI | As specified |
| UI | Streamlit | As specified, + `stpyvista` for 3D embedding |
| Data handling | Pandas | As specified |
| Deployment | Docker (+ docker-compose) | As specified |
| Version control | Git + GitHub | `.gitignore` provided |

**Flagged tech-stack issue:** "PyTorch + MONAI + nnU-Net" names two different frameworks as one
line. nnU-Net is a separate self-configuring package (`nnunetv2`) with its own CLI/experiment
structure; MONAI provides `DynUNet`, a MONAI-native architecture that follows nnU-Net's design
principles (dynamic kernel/stride configuration, deep supervision, instance norm) but stays
inside MONAI's Dataset/transform/training APIs. **Minimum modification applied:** this project
uses MONAI's `DynUNet` (see `src/segmentation/model.py`) so the whole pipeline is one consistent
codebase. Swapping in the standalone `nnunetv2` CLI later is possible without touching any code
downstream of the mask (measurement, reconstruction, matching, API) since both produce/consume
the same NIfTI label format.

## 5. System Architecture

```
Reference DB (CT + labels + implant catalog)          New patient CT
        │                                                    │
        ▼                                                    ▼
  CT Preprocessing (SimpleITK)                      CT Preprocessing (SimpleITK)
        │                                                    │
        ▼                                                    ▼
  [offline] nnU-Net-style training                  Femur/Tibia Segmentation (DynUNet)
  (src/segmentation/train.py)                                │
        │                                                    ▼
        │                                          Anatomical Feature Extraction
        │                                          (scikit-image + SciPy + NumPy)
        ▼                                                    │
  Implant catalog seeded into PostgreSQL  ◄───────────────────┘
  (src/database/seed_implant_catalog.py)            │
        │                                            ▼
        └──────────────────────────────►  Implant Size Matching (NumPy/SciPy nearest-neighbor)
                                                      │
                                                      ▼
                                          3D Reconstruction (PyVista) + Visualization
                                                      │
                                                      ▼
                                          FastAPI response → Streamlit UI → Surgeon Review
```

Detailed per-service tech breakdown was covered in the earlier architecture diagram in this
conversation; this README focuses on the ML/DL pipeline requested here.

## 6. Detailed Workflow

1. Clinician uploads a knee CT (NIfTI) via Streamlit.
2. Streamlit POSTs the file to `FastAPI: /patients/{id}/process`.
3. FastAPI saves it to a temp file, calls `segment_ct()`:
   a. `preprocess_ct()` — load → QC → reorient → denoise → resample → normalize.
   b. Trained DynUNet runs sliding-window inference → label mask (0/1/2).
4. `extract_anatomical_measurements()` computes femur/tibia ML width + AP dimension + volume.
5. `match_patient_to_implants()` queries PostgreSQL implant catalog, scores every candidate size
   by weighted Euclidean distance to the patient's measurements, flags overhang/under-coverage.
6. Result is persisted to `patient_measurements` and returned as JSON.
7. Streamlit renders the measurement table, recommended sizes, confidence badges, and warnings.
8. Surgeon reviews the output (and, optionally, the 3D mesh render) before finalizing the plan.

## 7. Dataset Requirements

- **Segmentation training data:** paired knee CT volumes + manually-annotated femur/tibia label
  masks. No dataset is bundled with this repo — you must source one. Candidates worth
  investigating (verify license/access terms yourself before use):
  - Institutional/hospital knee CT collections with IRB approval (most realistic for real
    accuracy numbers, but requires an institutional partnership).
  - Public musculoskeletal CT datasets that include lower-limb bone segmentation labels used in
    the literature reviewed earlier (e.g. the MOST Study–derived knee CT cohort referenced in
    published nnU-Net knee segmentation work) — access typically requires a data use agreement;
    do not assume open/unrestricted download.
  - **If no CT+label dataset is obtainable in your project timeline**, a documented, honest
    fallback is to prototype on a small set of manually-segmented CT scans you annotate yourselves
    (e.g. using 3D Slicer) — a few dozen cases lets you validate the *pipeline*, not a
    clinically-reliable model. State this limitation explicitly in any report/demo.
- **Implant catalog data:** `seed_implant_catalog.py` ships **illustrative placeholder
  dimensions only** — not real manufacturer specifications. Do not present these numbers as
  real without sourcing verified data from a manufacturer's public surgical technique guide.

## 8. Preprocessing Pipeline

Implemented in `src/preprocessing/ct_preprocessing.py`:
1. **Load** — DICOM series or NIfTI via SimpleITK.
2. **Quality control** — rejects degenerate volumes, localizer scans (<30 slices), NaNs, and
   implausibly narrow HU ranges (catches non-CT or pre-windowed exports before they reach the model).
3. **Standardize orientation** — reorient to a fixed anatomical frame (LPS) so downstream
   AP/ML axis assumptions in the measurement code hold across scanners.
4. **Denoise** — mild curvature-flow filtering (edge-preserving, unlike Gaussian blur).
5. **Resample** — isotropic 1.0mm spacing (configurable in `src/utils/config.py`).
6. **Normalize intensity** — clip to a bone-relevant HU window (-200 to 2000), scale to [0, 1].

**Augmentation (training only, `src/segmentation/dataset.py`):** random crop with
foreground oversampling, flips, 90° rotations, intensity scale/shift jitter, mild Gaussian
noise. Deliberately avoids elastic/nonlinear deformation, which would distort rigid bone
geometry the model should learn as geometrically consistent.

**Data splitting:** `create_splits()` splits at the **patient/case level** (not slice level) to
prevent data leakage — no case should have slices in both train and val/test.

## 9. Deep-Learning Model Selection

**Segmentation: 3D DynUNet (MONAI's nnU-Net-style architecture).** Rationale, grounded in the
literature reviewed earlier in this conversation: nnU-Net/DynUNet-class models are the
consistently validated choice for this exact task (knee CT femur/tibia/patella segmentation),
and 3D approaches were repeatedly shown to outperform 2D on lower-limb CT segmentation. Instance
normalization, deep supervision, and a poly-LR schedule follow nnU-Net's default recipe, which
is documented to generalize well without heavy manual tuning — important for a student/hackathon
timeline without resources for extensive architecture search.

**Why not 2D:** CT bone segmentation benefits from full 3D context (femur/tibia are long,
continuous structures; 2D per-slice models are prone to discontinuities and confusing femur vs.
tibia at the joint line, a failure mode explicitly documented in the two-stage segmentation
literature reviewed earlier).

**Implant size matching: deterministic geometric nearest-neighbor matching (NumPy/SciPy), not a
second deep-learning model.** This is a request for tabular geometric comparison against a
structured catalog, which is exactly what the specified tech stack (NumPy + SciPy for matching
and ranking) calls for. See `src/matching/implant_matcher.py` docstring for the full rationale.
A future ML calibration layer (e.g. gradient-boosted trees on real outcome labels) is a
documented extension, not part of this basic version (no outcome labels available).

## 10. Training Pipeline

See `src/segmentation/train.py`:
- **Loss:** DiceCE (Dice + cross-entropy) — standard nnU-Net-style choice, robust to the femur/
  tibia-vs-background class imbalance better than CE alone.
- **Optimizer:** SGD, Nesterov momentum 0.99, weight decay 3e-5.
- **LR schedule:** polynomial decay (`poly_lr`), nnU-Net default.
- **Batch size:** 2 (3D patches are memory-heavy) with 4-step gradient accumulation → effective
  batch size 8. Adjust `TRAIN_BATCH_SIZE` in `src/utils/config.py` to fit your GPU.
- **Epochs:** up to 300, with early stopping (patience 40 epochs without val Dice improvement).
- **Class imbalance handling:** foreground-oversampled random cropping (`RandCropByPosNegLabeld`,
  pos:neg = 2:1) + DiceCE loss.
- **Overfitting / limited data handling:** anatomically-conservative augmentation, deep
  supervision (regularizing auxiliary losses), early stopping, and — most importantly — an
  honest dataset-size caveat (see Limitations).
- **Data leakage prevention:** case-level splitting (`create_splits`), never slice-level.

## 11. Evaluation Metrics

- **Segmentation:** Dice Similarity Coefficient per class (femur, tibia), computed via
  `monai.metrics.DiceMetric` in `train.py`'s `validate()`. Add Hausdorff distance and average
  surface distance if boundary precision needs closer QA (not included in this basic version —
  straightforward to add via `monai.metrics.HausdorffDistanceMetric`).
- **Measurement accuracy:** compare extracted ML/AP dimensions against a manually-measured
  ground truth subset (not automatable — a person with calipers/software must produce the
  reference values). Not fabricated here; this is a "not yet evaluated" item until real data exists.
- **Implant matching:** exact-size accuracy and within-±1-size accuracy **can only be computed
  once you have real historical implanted-size outcomes** to compare predictions against — this
  prototype's matching engine is validated for *internal logical correctness* (see
  `tests/test_measurement_and_matching.py`, which passes), not yet for *clinical accuracy*.

## 12. Inference Pipeline

`src/segmentation/infer.py::segment_ct()`:
`preprocess_ct()` → tensor → `sliding_window_inference()` (tiles the full volume using the
training patch size, stitches results) → `argmax` → label mask. Exposed end-to-end through
`FastAPI: POST /patients/{id}/process`, which also runs measurement + matching + persistence.

## 13. Project Folder Structure

```
knee-implant-sizing/
├── data/
│   ├── raw/            # original DICOM/NIfTI (gitignored - never commit real patient data)
│   ├── processed/      # preprocessed image.nii.gz + label.nii.gz per case
│   └── splits/         # train.json / val.json / test.json (case ID lists)
├── src/
│   ├── preprocessing/  # ct_preprocessing.py
│   ├── segmentation/   # model.py, dataset.py, train.py, infer.py
│   ├── measurement/    # anatomical_measurement.py
│   ├── reconstruction/ # mesh_reconstruction.py
│   ├── matching/       # implant_matcher.py
│   ├── database/       # models.py, seed_implant_catalog.py
│   ├── api/            # main.py, schemas.py
│   └── utils/          # config.py
├── app/
│   └── streamlit_app.py
├── tests/
│   └── test_measurement_and_matching.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## 14. Implementation Roadmap

1. **Data acquisition** — source or annotate a CT+label dataset (see Section 7). *Blocking step.*
2. **Preprocessing validation** — run `ct_preprocessing.py` on a handful of real scans, sanity-check
   HU ranges/orientation visually (e.g. in 3D Slicer).
3. **Segmentation training** — `create_splits()` → `train.py`; monitor val Dice; expect several
   hours to ~1 day on a single modern GPU for a few hundred cases (see Section 16).
4. **Measurement validation** — spot-check `extract_anatomical_measurements()` output against a
   few manually-measured cases.
5. **Implant catalog** — replace `seed_implant_catalog.py` placeholders with a real, licensed/
   public size chart before any demo implying real clinical numbers.
6. **API + UI integration** — `uvicorn src.api.main:app`, `streamlit run app/streamlit_app.py`.
7. **Dockerize** — `docker compose -f docker/docker-compose.yml up --build`.
8. **Surgeon-facing review pass** — get a clinical collaborator (if available) to sanity-check
   outputs before presenting matching scores as meaningful.

## 15. Required Python Libraries

See `requirements.txt`. Core: `torch`, `monai`, `SimpleITK`, `numpy`, `scipy`, `scikit-image`,
`pandas`, `pyvista`, `vtk`, `stpyvista`, `fastapi`, `uvicorn`, `SQLAlchemy`, `psycopg2-binary`,
`streamlit`, `pytest`.

## 16. Hardware/GPU Requirements

- **Training:** an NVIDIA GPU with **≥12GB VRAM** is realistic for the default `PATCH_SIZE =
  (96, 160, 160)` at batch size 2. Reduce patch size or batch size for smaller GPUs (e.g. a free
  Colab T4). Training on CPU only is technically possible but impractically slow for 3D
  volumes (likely many days) — the code will run on CPU (device auto-detected in `train.py`)
  but this is not a realistic path to a usable model.
- **Inference:** a single forward pass with sliding-window inference is far cheaper than
  training; a mid-range GPU (or patient GPU-less CPU inference, at maybe 1-5 minutes per scan)
  is workable for a demo.
- **Everything else** (measurement, matching, API, UI, PostgreSQL) is CPU-only and lightweight.

## 17. Expected Outputs

- Label mask (`.nii.gz`, values 0/1/2) per processed CT.
- Femur/tibia ML width, AP dimension, volume (mm/mm³).
- Ranked implant size candidates per component (femoral, tibial) with matching score,
  overhang/under-coverage flags, and a confidence label.
- Optional 3D surface meshes (`.vtp`) and PNG preview render for visual QC.
- All of the above persisted to PostgreSQL and viewable via the Streamlit UI.

## 18. Limitations & Solutions

| Limitation | Status | Solution / mitigation |
|---|---|---|
| No real CT+label dataset bundled | Blocking for real accuracy | Source institutional data or annotate a small pilot set (Section 7); state pilot-scale caveats explicitly |
| Implant catalog uses placeholder dimensions | Must fix before any real demo | Replace with verified manufacturer data before presenting real numbers |
| AP/ML measurement is a bounding-box style measurement, not full landmark-based surgical planning (epicondylar axis etc.) | Documented, not hidden | Acceptable for a prototype/decision-support tool; a landmark-detection model is a distinct, larger future project |
| No historical implanted-size outcome data | Blocks ML calibration layer | Deterministic geometric matching is the primary engine; add XGBoost/LightGBM calibration once outcome data exists |
| No clinical validation performed | This is a prototype | Do not present matching scores as clinically validated accuracy; needs surgeon review + a real validation study before any clinical claim |
| Small-dataset overfitting risk | Mitigated, not eliminated | Augmentation, early stopping, deep supervision (Section 10); still expect wide confidence intervals on a small pilot dataset |
| Orientation-dependent AP/ML axis assumption | Documented | Preprocessing enforces a fixed LPS orientation; verify this holds for your actual scanner/protocol before trusting axis labels |

## 19. Future Improvements

- Landmark-based measurement (transepicondylar axis, posterior condylar axis) to match
  commercial surgical planning software more closely.
- ML calibration layer (gradient-boosted trees) once real implanted-size outcomes are available.
- Vendor-abstracted implant catalog supporting multiple manufacturers/systems simultaneously.
- Interactive segmentation-correction UI (adjust boundary, see measurements recompute live).
- Hausdorff distance / average surface distance added to segmentation evaluation.
- Swap DynUNet for the standalone `nnunetv2` CLI if automatic cross-validation ensembling is needed.

## 20. Final End-to-End Pipeline (summary)

```
CT (DICOM/NIfTI)
  → SimpleITK preprocessing (QC, reorient, denoise, resample, normalize)
  → MONAI DynUNet segmentation (femur, tibia)
  → scikit-image/SciPy/NumPy anatomical measurement (ML width, AP dimension, volume)
  → NumPy/SciPy nearest-neighbor implant size matching against PostgreSQL catalog
  → PyVista 3D mesh reconstruction (visual QC)
  → FastAPI response → Streamlit UI
  → Surgeon review (final decision authority)
```

---

### Quick start (once you have real data)

```bash
pip install -r requirements.txt

# 1. Prepare data/processed/<case_id>/{image,label}.nii.gz for each case, then:
python -c "from src.segmentation.dataset import create_splits; create_splits(['case_0001', 'case_0002', ...])"

# 2. Train
python -m src.segmentation.train

# 3. Seed implant catalog (replace placeholder data first!)
python -m src.database.seed_implant_catalog

# 4. Run backend + UI
uvicorn src.api.main:app --reload
streamlit run app/streamlit_app.py

# or, containerized:
docker compose -f docker/docker-compose.yml up --build
```

### Running tests (no GPU/model/Postgres required)

```bash
pip install pytest numpy scipy scikit-image SimpleITK sqlalchemy psycopg2-binary
PYTHONPATH=. pytest tests/ -v
```
`tests/test_measurement_and_matching.py` (4 tests) has been verified to pass against this
codebase — it validates the measurement geometry and matching/scoring logic independent of
having a trained segmentation model or a running PostgreSQL instance.
