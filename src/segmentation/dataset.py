"""
Dataset & DataLoader construction for femur/tibia CT segmentation.

Expects data already produced by src/preprocessing/ct_preprocessing.py
and organized as described in the project README:

    data/processed/
        case_0001/
            image.nii.gz
            label.nii.gz      # 0=background, 1=femur, 2=tibia
        case_0002/
            ...

    data/splits/
        train.json   # ["case_0001", "case_0003", ...]
        val.json
        test.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,

    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ToTensord,
)

from src.utils.config import DATA_PROCESSED_DIR, DATA_SPLITS_DIR, PATCH_SIZE, TARGET_SPACING


def _load_case_ids(split_name: str) -> List[str]:
    split_path = DATA_SPLITS_DIR / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file {split_path} not found. Run "
            f"scripts to generate train/val/test splits before training."
        )
    with open(split_path) as f:
        return json.load(f)


def _build_data_dicts(case_ids: List[str]) -> List[dict]:
    data = []
    for case_id in case_ids:
        case_dir = DATA_PROCESSED_DIR / case_id
        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "label.nii.gz"
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing image/label for case {case_id} in {case_dir}")
        data.append({"image": str(image_path), "label": str(label_path)})
    return data


from monai.data import CacheDataset, DataLoader

def get_train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING,
                      mode=("bilinear", "nearest")),
            DivisiblePadd(keys=["image", "label"], k=16),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=2, neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms() -> Compose:
    """No augmentation for validation/test - deterministic preprocessing only."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=TARGET_SPACING,
                      mode=("bilinear", "nearest")),
            DivisiblePadd(keys=["image", "label"], k=16),
            ToTensord(keys=["image", "label"]),
        ]
    )





from monai.data import CacheDataset, DataLoader, pad_list_data_collate

def get_dataloader(split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    split: "train" | "val" | "test"
    """
    case_ids = _load_case_ids(split)
    data_dicts = _build_data_dicts(case_ids)
    transforms = get_train_transforms() if split == "train" else get_val_transforms()

    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.5, num_workers=num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=pad_list_data_collate,
        pin_memory=True,
    )



def create_splits(case_ids: List[str], train_frac=0.7, val_frac=0.15, seed: int = 42) -> None:
    """
    Utility to generate train/val/test split JSON files from a full list
    of case IDs. Splitting is done at the *patient/case* level (not slice
    level) to avoid data leakage between splits.
    """
    import random

    rng = random.Random(seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        splits = {
            "train": shuffled,
            "val": shuffled,
            "test": shuffled,
        }
    else:
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:] if (n_train + n_val) < n else shuffled[n_train:],
        }

    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        with open(DATA_SPLITS_DIR / f"{name}.json", "w") as f:
            json.dump(ids, f, indent=2)
        print(f"{name}: {len(ids)} cases")

