"""
Segmentation model definition.

TECH STACK NOTE (flagged per task instructions):
--------------------------------------------------
The requested stack lists "PyTorch + MONAI + nnU-Net" as one line. In
practice these are two different things:

    * nnU-Net (github.com/MIC-DKFZ/nnUNet) is a *separate*, self-configuring
      framework/CLI (package `nnunetv2`) that owns its own preprocessing,
      training loop, and experiment folder structure. Using it means
      largely stepping outside MONAI's APIs.
    * MONAI ships `DynUNet`, which is a MONAI-native reimplementation of
      the nnU-Net architecture (dynamic kernel/stride configuration based
      on patch size and spacing) that plugs directly into MONAI
      Dataset/DataLoader/transforms/training loops.

Minimum modification: this project uses **MONAI's DynUNet** so the
architecture, data pipeline, augmentation, and training loop all stay
inside one consistent MONAI-based codebase (simpler, more reproducible
for a student project). If you specifically need nnU-Net's automatic
pipeline configuration and cross-validation ensembling later, swap this
module for a call into the `nnunetv2` CLI - the rest of the pipeline
(preprocessing output format, measurement/matching code) is unaffected
because both approaches consume/produce NIfTI volumes and label masks.
"""
from __future__ import annotations

from monai.networks.nets import UNet

from src.utils.config import NUM_CLASSES, PATCH_SIZE



def build_segmentation_model(
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    patch_size=PATCH_SIZE,
) -> UNet:
    """
    Build a 3D U-Net configured for femur/tibia/background segmentation.
    Uses anisotropic strides (1, 2, 2) to handle joint CT volumes cleanly.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2)),
        num_res_units=2,
        norm="instance",
    )
    return model




def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_segmentation_model()
    dummy = torch.randn(1, 1, *PATCH_SIZE)
    with torch.no_grad():
        out = m(dummy)
    # deep_supervision=True returns a stacked tensor of shape
    # (B, num_supervision_heads, C, D, H, W) during training;
    # only out[:, 0] (the full-resolution head) is used at inference.
    print(f"Parameters: {count_parameters(m):,}")
    print(f"Output shape: {out.shape if not isinstance(out, (list, tuple)) else [o.shape for o in out]}")
