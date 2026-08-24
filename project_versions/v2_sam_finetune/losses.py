import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        return 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()

class JointMaskClassLoss(nn.Module):
    def __init__(self, lambda_mask=1.0, lambda_class=0.5, dice_weight=0.5, bce_weight=0.5, class_weights=None):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_class = lambda_class
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        if class_weights is None:
            weights = torch.tensor([1.0, 3.0, 1.5, 8.0, 6.0, 1.0, 5.0, 0.5])
            class_weights = weights / weights.sum() * len(weights)
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, mask_logits, class_logits, mask_target, class_target):
        if mask_target.shape[2:] != mask_logits.shape[2:]:
            mask_target = F.interpolate(mask_target, size=mask_logits.shape[2:], mode="bilinear", align_corners=False)
        dice_loss = self.dice(mask_logits, mask_target)
        bce_loss = self.bce(mask_logits, mask_target)
        mask_loss = 0.5 * dice_loss + 0.5 * bce_loss
        class_loss = self.ce(class_logits, class_target)
        total = self.lambda_mask * mask_loss + self.lambda_class * class_loss
        return total, {"loss": total, "mask_loss": mask_loss, "class_loss": class_loss, "dice_loss": dice_loss, "bce_loss": bce_loss}