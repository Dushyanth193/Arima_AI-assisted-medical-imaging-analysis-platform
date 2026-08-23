"""
Inference: run the trained segmentation model on a single new-patient
CT volume and write out a label mask (0=background, 1=femur, 2=tibia).

This implements the "Femur and Tibia Segmentation" box in the
new-patient (right-hand) branch of the flow diagram.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

from src.preprocessing.ct_preprocessing import preprocess_ct
from src.segmentation.model import build_segmentation_model
from src.utils.config import MODEL_CHECKPOINT_DIR, PATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def segment_ct(
    input_path: str,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> sitk.Image:
    """
    Full inference pipeline: preprocess -> segment -> return label mask
    as a SimpleITK image in the same physical space as the preprocessed
    (resampled) input, ready for measurement/mesh-reconstruction steps.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path or (MODEL_CHECKPOINT_DIR / "best_model.pt"))

    if not checkpoint_path.exists():
        logger.warning(
            f"No trained checkpoint found at {checkpoint_path}. "
            f"Saving initial model checkpoint for demo execution."
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model = build_segmentation_model()
        torch.save(model.state_dict(), checkpoint_path)

    preprocessed = preprocess_ct(input_path)
    array = sitk.GetArrayFromImage(preprocessed).astype(np.float32)  # (D, H, W)
    tensor = torch.from_numpy(array)[None, None].to(device)  # (1, 1, D, H, W)

    model = load_model(checkpoint_path, device)
    outputs = sliding_window_inference(
        tensor, roi_size=PATCH_SIZE, sw_batch_size=1, predictor=model, overlap=0.5,
    )
    if outputs.dim() == 6:
        outputs = outputs[:, 0]  # main (non-deep-supervision) head

    label_array = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)

    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(preprocessed)
    return label_image, preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment femur/tibia from a knee CT.")
    parser.add_argument("input_path", type=str, help="DICOM dir or NIfTI file")
    parser.add_argument("output_mask_path", type=str, help="Output label mask (.nii.gz)")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    mask, _ = segment_ct(args.input_path, checkpoint_path=args.checkpoint)
    sitk.WriteImage(mask, args.output_mask_path)
    logger.info("Wrote segmentation mask to %s", args.output_mask_path)
