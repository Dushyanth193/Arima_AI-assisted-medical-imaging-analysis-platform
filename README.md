AI-Assisted Medical Imaging Analysis Platform
A Tool for Orthopedic Surgeons
Team: ARIMA
---
Table of Contents
What This Project Is
The Problem
Our Solution
How It Works (Workflow)
Tech Stack
Example Output
References
---
What This Project Is
An AI-assisted decision-support platform that analyzes knee MRI and CT scans to help orthopedic surgeons with two tasks:
Task	Question it answers
OA Assessment	"How healthy is this patient's meniscus, and are they showing signs of osteoarthritis?"
Implant Planning	"What implant size best fits this patient's bone anatomy for knee replacement surgery?"
The platform does not replace the surgeon — it produces objective, quantitative measurements and ranked recommendations that the surgeon reviews before making the final call.
---
The Problem
Problem A — Meniscus Analysis is Manual and Inconsistent
The medial meniscus absorbs shock and stabilizes the knee joint. Damage to it is closely tied to OA progression. Today, radiologists assess it by eye:
Slow — manual review takes time per patient
Subjective — different radiologists reach different conclusions
Incomplete — there's no standard way to combine meniscus thickness with a patient's age, sex, and OA status into one consistent score
Problem B — Implant Sizing is Manual and Variable
Total Knee Arthroplasty (TKA) requires an implant that matches the patient's exact bone shape. Today, sizing depends on:
Manual measurement on 2D X-rays (limited depth/3D information)
Surgeon experience, which varies from case to case
Single-hospital data, which doesn't generalize well across diverse patients
A poorly sized implant can cause overhang, soft-tissue irritation, and lower long-term patient satisfaction.
---
Our Solution
	Problem A: OA Assessment	Problem B: Implant Sizing
Input	Knee MRI	Knee CT
What AI does	Segments the medial meniscus and measures volume, thickness, and extrusion	Segments the femur and tibia and measures width/AP dimensions
How it decides	Compares the patient's features (+ age, sex, BMI) against a reference database to classify OA	Compares the patient's measurements against an implant-size database and ranks the best-fitting options
Output	OA classification + probability + measurements	Ranked implant size recommendations
Both outputs are combined into a single Integrated AI Report for the surgeon to review.
---
How It Works (Workflow)
```
                         KNEE IMAGING DATA
                  (MRI for meniscus · CT for implants)
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            MODULE 1: MRI                 MODULE 2: CT
       (Meniscus & OA Classification)   (Implant Sizing)
                    │                             │
      1. Preprocess reference + new MRI   1. Preprocess reference + new CT
      2. AI segments medial meniscus      2. AI segments femur & tibia
      3. Extract volume/thickness/        3. Extract width/AP measurements
         extrusion features
      4. OA Classification Model          4. Implant Size Matching Model
         compares to reference DB            compares to implant DB
                    │                             │
                    ▼                             ▼
          OA Assessment Output          Implant Sizing Output
        (classification, probability,   (recommended sizes,
         meniscus measurements)          alignment)
                    └──────────────┬──────────────┘
                                   ▼
                        INTEGRATED AI REPORT
              (classifications, sizes, alternatives, visuals)
                                   ▼
                     CLINICIAN / SURGEON REVIEW
                    (AI is decision support only —
                     final call stays with the surgeon)
```
---
Tech Stack
Grouped by pipeline stage:
1. Image Input & Preprocessing
`DICOM / NIfTI` — input MRI/CT format
`SimpleITK` — load, preprocess, spacing & orientation; convert voxel measurements → mm
`NumPy` — image/mask numerical processing
2. AI Segmentation & Classification
`PyTorch + MONAI` — medical AI framework
`nnU-Net` — segments meniscus, femur & tibia
`scikit-image` — mask cleaning & boundary extraction
`scikit-learn` — OA classification
3. Measurement & Matching
`SciPy + NumPy` — geometry, anatomical measurements, implant size matching & ranking
`PostgreSQL` — implant specification database
`Pandas` — organizes measurements/results
4. Visualization & Delivery
`PyVista / VTK` — 3D anatomy visualization
`Plotly` — result/measurement charts
`Streamlit` — final application/dashboard
---
Example Output
```
OA ASSESSMENT                          IMPLANT SIZING
─────────────────                      ─────────────────
OA Detected                            Femoral Component: Size 6
Probability: 85%                       Tibial Baseplate: Size 5
Meniscus Volume: 1.2 cm³               Insert Thickness: 10 mm
Meniscus Thickness: 4.5 mm             Implant Alignment: Neutral (0°)
Meniscus Extrusion: 3.2 mm
```
---
References
Problem Research
Osteoarthritis — StatPearls (NCBI Bookshelf): https://www.ncbi.nlm.nih.gov/books/NBK618758/
Osteoarthritis — WHO Fact Sheet: https://www.who.int/news-room/fact-sheets/detail/osteoarthritis
Existing Solutions
Existing Solution Overview (reference document): https://drive.google.com/file/d/11PjL8Eb7F5bt1hFv5vqdbSq5VpWLkEIC/view?usp=sharing
Related Study (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC9206408/
