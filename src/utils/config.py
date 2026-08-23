"""
Central configuration for the knee CT implant-sizing pipeline.

Every other module imports paths/constants from here instead of
hardcoding them, so the whole project has one place to change
data locations, spacing targets, label IDs, etc.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"          # original DICOM/NIfTI CT volumes
PATIENT_DATASET_DIR = PROJECT_ROOT / "Patient_Dataset" # multi-patient DICOM dataset directory
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # resampled/normalized NIfTI + masks
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"    # train/val/test CSV or JSON split files


MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True)

INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"
INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Image preprocessing constants
# ---------------------------------------------------------------------------
# Target isotropic spacing (mm) after resampling. Knee CT is usually
# already close to isotropic (~0.3-1.0mm in-plane); 1.0mm isotropic is a
# reasonable balance of anatomical detail vs. GPU memory for training.
TARGET_SPACING = (1.0, 1.0, 1.0)

# Hounsfield Unit clipping window for bone. Cortical bone is roughly
# 300-1900 HU; soft tissue/air/metal artifact outside that range are not
# informative for femur/tibia segmentation and are clipped to reduce
# the input dynamic range the network has to learn.
HU_CLIP_MIN = -200
HU_CLIP_MAX = 2000

# Fixed-size crop/patch used for training (D, H, W) in voxels after
# resampling. Chosen to comfortably fit femur+tibia around the joint
# line on a single consumer/workstation GPU (>=12GB VRAM) with MONAI's
# sliding-window inference for anything larger at inference time.
PATCH_SIZE = (16, 64, 64)


# ---------------------------------------------------------------------------
# Segmentation labels
# ---------------------------------------------------------------------------
LABELS = {
    "background": 0,
    "femur": 1,
    "tibia": 2,
}
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 2          # 3D patches are memory-heavy; small batch + grad accumulation
GRAD_ACCUMULATION_STEPS = 4   # effective batch size = 2 * 4 = 8
NUM_EPOCHS = 15

LEARNING_RATE = 1e-2          # SGD-with-momentum, nnU-Net-style poly LR schedule
WEIGHT_DECAY = 3e-5
VAL_INTERVAL = 5              # run validation every N epochs
EARLY_STOPPING_PATIENCE = 40  # epochs without val Dice improvement before stopping

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
# Overridden by environment variables in real deployments (see docker-compose.yml).
import os  # noqa: E402

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///knee_implant.db",
)
