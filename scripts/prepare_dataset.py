"""
Prepare Dataset Script
Discovers cases and generates patient-level train/val/test split files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import create_splits
from src.utils.config import DATA_PROCESSED_DIR



def main():
    parser = argparse.ArgumentParser(description="Create patient-level data splits.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_PROCESSED_DIR))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Data directory {data_path} does not exist. Creating...")
        data_path.mkdir(parents=True, exist_ok=True)

    case_ids = [d.name for d in data_path.iterdir() if d.is_dir()]
    if not case_ids:
        print(f"No processed case directories found in {data_path}.")
        return

    val_frac = args.val_ratio
    train_frac = 1.0 - args.val_ratio - args.test_ratio
    create_splits(case_ids, train_frac=train_frac, val_frac=val_frac)
    print(f"Created dataset splits for {len(case_ids)} cases.")



if __name__ == "__main__":
    main()
