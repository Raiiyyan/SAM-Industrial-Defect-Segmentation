"""
Joint loss function for IndustrialSAM.

Combines Dice + BCE for mask segmentation and CrossEntropy for 8-class
defect classification.  Automatically resizes ground-truth masks to match
model output resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """Soft Dice loss for binary mask segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class JointMaskClassLoss(nn.Module):
    """Weighted sum of mask segmentation loss and class classification loss.

    Parameters
    ----------
    lambda_mask : float  Weight for the combined mask loss (default 1.0).
    lambda_class : float  Weight for the class loss (default 0.5).
    dice_weight : float  Weight of Dice within the mask loss (default 0.5).
    bce_weight : float  Weight of BCE within the mask loss (default 0.5).
    class_weights : Tensor or None  Per-class weights for CrossEntropyLoss.
        If None, uses inverse-frequency weights to handle class imbalance.

    Returns
    -------
    total : Tensor  Scalar weighted loss (for backprop).
    dict  : {"loss", "mask_loss", "class_loss", "dice_loss", "bce_loss"}
    """

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_class: float = 0.5,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_class = lambda_class
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        # Inverse-frequency weights to handle class imbalance
        if class_weights is None:
            # Based on observed distribution: 0:0, 1:67, 2:143, 3:14, 4:20, 5:0, 6:27, 7:329
            weights = torch.tensor([1.0, 3.0, 1.5, 8.0, 6.0, 1.0, 5.0, 0.5])
            class_weights = weights / weights.sum() * len(weights)  # normalize
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        mask_target: torch.Tensor,
        class_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Resize target to match prediction if needed (resolution-dependent mask output)
        if mask_target.shape[2:] != mask_logits.shape[2:]:
            mask_target = F.interpolate(
                mask_target,
                size=mask_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        dice_loss = self.dice(mask_logits, mask_target)
        bce_loss = self.bce(mask_logits, mask_target)
        mask_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        class_loss = self.ce(class_logits, class_target)
        total = self.lambda_mask * mask_loss + self.lambda_class * class_loss
        return total, {
            "loss": total,
            "mask_loss": mask_loss,
            "class_loss": class_loss,
            "dice_loss": dice_loss,
            "bce_loss": bce_loss,
        }
