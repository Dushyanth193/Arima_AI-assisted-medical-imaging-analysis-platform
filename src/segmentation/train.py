"""
Training loop for femur/tibia CT segmentation (DynUNet).

Loss:      DiceCE (Dice + cross-entropy) - Dice alone can plateau on
           small/thin structures early in training; combining with CE
           stabilizes early gradients. Standard nnU-Net-style choice.
Optimizer: SGD with Nesterov momentum + polynomial LR decay, matching
           nnU-Net's default recipe, which is known to be robust across
           medical segmentation tasks without heavy tuning.
Imbalance: femur/tibia occupy a small fraction of the CT volume relative
           to background. Handled via (a) RandCropByPosNegLabeld
           oversampling foreground patches during training (see
           dataset.py), and (b) DiceCE loss, which is far less sensitive
           to class imbalance than plain pixel-wise cross-entropy alone.
Overfitting / limited data: heavy but anatomically-plausible augmentation
           (dataset.py), early stopping on validation Dice, and deep
           supervision (auxiliary losses at intermediate decoder scales)
           which regularizes training on small datasets.
"""
from __future__ import annotations

import logging
import time

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

from src.segmentation.dataset import get_dataloader
from src.segmentation.model import build_segmentation_model
from src.utils.config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODEL_CHECKPOINT_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    PATCH_SIZE,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    VAL_INTERVAL,
    WEIGHT_DECAY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def poly_lr(epoch: int, max_epochs: int, initial_lr: float, exponent: float = 0.9) -> float:
    """nnU-Net's polynomial LR decay schedule."""
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def train(resume_checkpoint: str | None = None) -> None:
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning(
            "No CUDA GPU detected - training 3D volumes on CPU is impractically "
            "slow. See README 'Hardware/GPU Requirements' before running a real job."
        )

    model = build_segmentation_model().to(device)
    if resume_checkpoint:
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        logger.info("Resumed weights from %s", resume_checkpoint)

    train_loader = get_dataloader("train", batch_size=TRAIN_BATCH_SIZE)
    val_loader = get_dataloader("val", batch_size=1)  # full-volume sliding-window inference at val time

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.99,
        nesterov=True, weight_decay=WEIGHT_DECAY,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    best_val_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        lr = poly_lr(epoch, NUM_EPOCHS, LEARNING_RATE)
        for g in optimizer.param_groups:
            g["lr"] = lr

        epoch_loss = 0.0
        start = time.time()
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # DynUNet with deep_supervision=True returns shape
            # (B, num_heads, C, D, H, W); compute loss at the main head
            # plus averaged auxiliary heads, weighted by decreasing importance.
            if outputs.dim() == 6:
                main_out = outputs[:, 0]
                loss = loss_fn(main_out, labels)
                for i in range(1, outputs.shape[1]):
                    loss = loss + 0.5 ** i * loss_fn(outputs[:, i], labels)
            else:
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        logger.info(
            "Epoch %d/%d - loss=%.4f - lr=%.6f - %.1fs",
            epoch + 1, NUM_EPOCHS, epoch_loss, lr, time.time() - start,
        )

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_dice = validate(model, val_loader, device, dice_metric, post_pred, post_label)
            logger.info("Epoch %d - validation mean Dice (femur, tibia): %s", epoch + 1, val_dice)

            mean_dice = float(val_dice.mean())
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                epochs_without_improvement = 0
                ckpt_path = MODEL_CHECKPOINT_DIR / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                logger.info("New best val Dice %.4f - saved to %s", best_val_dice, ckpt_path)
            else:
                epochs_without_improvement += VAL_INTERVAL

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "No val Dice improvement for %d epochs - stopping early at epoch %d.",
                    epochs_without_improvement, epoch + 1,
                )
                break

    final_ckpt = MODEL_CHECKPOINT_DIR / "final_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.info("Training complete. Final weights saved to %s", final_ckpt)


@torch.no_grad()
def validate(model, val_loader, device, dice_metric, post_pred, post_label):
    model.eval()
    dice_metric.reset()
    for batch in val_loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        # Full CT volumes don't fit in memory at once for 3D inference;
        # sliding_window_inference tiles the volume using the same patch
        # size training used, and stitches predictions back together.
        outputs = sliding_window_inference(
            images, roi_size=PATCH_SIZE, sw_batch_size=1, predictor=model, overlap=0.5,
        )
        if outputs.dim() == 6:  # deep supervision output at inference - use main head only
            outputs = outputs[:, 0]

        outputs_list = [post_pred(o) for o in torch.unbind(outputs, dim=0)]
        labels_list = [post_label(l) for l in torch.unbind(labels, dim=0)]
        dice_metric(y_pred=outputs_list, y=labels_list)

    return dice_metric.aggregate()


if __name__ == "__main__":
    train()
