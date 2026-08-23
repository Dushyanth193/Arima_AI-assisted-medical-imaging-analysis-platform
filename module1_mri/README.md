# Module 1 — MRI Meniscus Analysis & OA Classification

Basic, working implementation of the left-hand branch of your workflow diagram:
Reference MRI DB → Preprocessing → Meniscus Segmentation → Feature Extraction →
OA Classification → OA Assessment Output.

## What's real vs. placeholder

| Component | Status |
|---|---|
| I/O (`src/io_utils.py`) | Real — SimpleITK load/save |
| Preprocessing (`src/preprocessing.py`) | Real — resample, denoise, N4, z-score normalize |
| Feature extraction (`src/features.py`) | Real — volume/thickness validated against an analytical sphere (see `tests/test_features.py`) |
| OA classifier (`src/classification.py`) | Real — Random Forest baseline, trainable on your reference DB |
| **Segmentation** (`src/segmentation.py`) | `segment_with_nnunet()` is the real production wrapper around nnU-Net, but it needs **your trained weights** to run. `segment_meniscus_placeholder()` is a simple intensity/morphology stand-in so the rest of the pipeline is testable *today*, before nnU-Net is trained. **Swap this out — it is not clinically meaningful.** |

## Setup

```bash
pip install -r requirements.txt
```

`SimpleITK` and `nnunetv2` weren't available in the sandbox this was built in
(no network access there) — the code is written against their documented
APIs but hasn't been execution-tested against real `.nii.gz` files or a
trained nnU-Net model. Everything downstream of segmentation (features,
classifier, reporting) **has** been tested here using synthetic data — see
`tests/` and `demo_end_to_end.py`.

## Try it right now (no real data needed)

```bash
python demo_end_to_end.py
```

Runs the full chain on a synthetic volume: placeholder segmentation →
feature extraction → classifier training on synthetic reference data →
prediction → printed report.

## Run tests

```bash
python tests/test_features.py        # validates volume/thickness math
python tests/test_classification.py  # validates classifier trains + predicts sensibly
```

## Using it on real data

1. Build `data/reference_db/labels.csv` with columns:
   `subject_id, image_path, age, sex, bmi, oa_label`
2. Train the reference feature table + classifier:
   ```python
   from src.pipeline import build_reference_database
   from src.classification import train_oa_classifier

   ref_features = build_reference_database("data/reference_db/labels.csv")
   clf, metrics = train_oa_classifier(ref_features, model_path="models/oa_classifier.joblib")
   print(metrics)
   ```
3. Once nnU-Net is trained, swap `segment_meniscus_placeholder` for
   `segment_with_nnunet` inside `src/pipeline.py`'s `_process_single_case()`.
4. Run a new patient:
   ```python
   from src.pipeline import run_module1
   from src.classification import load_classifier

   clf = load_classifier("models/oa_classifier.joblib")
   result = run_module1("data/new_patient/case001.nii.gz", age=62, sex="F", bmi=28, clf=clf)
   print(result["report"])
   ```

## Next steps / known gaps (see implementation plan for detail)

- **Extrusion** needs a tibia mask as a landmark — not wired up to a real
  bone segmenter yet; `compute_extrusion_mm()` works once you pass one in.
- Replace the placeholder segmenter with trained nnU-Net as soon as you have
  annotated data — this is the actual critical path for clinical validity.
- Confirm the `axis` parameter in `compute_extrusion_mm()` matches your
  actual scan orientation (medial-lateral direction) before trusting the
  extrusion number.
