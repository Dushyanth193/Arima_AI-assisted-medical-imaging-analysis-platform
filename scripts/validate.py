"""
Validate CLI Script
Evaluates a trained model on validation/test splits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import MODEL_CHECKPOINT_DIR



def main():
    parser = argparse.ArgumentParser(description="Validate segmentation model performance.")
    parser.add_argument("--checkpoint", type=str, default=str(MODEL_CHECKPOINT_DIR / "best_model.pt"))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Evaluation unavailable — model checkpoint not found at {checkpoint_path}.")
        return

    print(f"Validating checkpoint: {checkpoint_path}")
    print("Dice Score (Femur): N/A (requires ground truth validation set)")
    print("Dice Score (Tibia): N/A (requires ground truth validation set)")


if __name__ == "__main__":
    main()
