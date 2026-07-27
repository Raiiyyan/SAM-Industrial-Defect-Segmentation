"""Training pipeline for IndustrialSAM — joint mask + class prediction.

Trains on all 6 industrial-defect datasets with a combined
Dice + BCE (mask) and CrossEntropy (class) loss.

Usage::

    python train.py --checkpoint sam_vit_b_01ec64.pth --root-dir G:/Dataset
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from dataset import UniversalIndustrialDataset, SUPPORTED_DATASETS
from label_mapping import map_dataset_item
from model import IndustrialSAM
from model_setup import _ADAPTER_PREFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ===================================================================
# 1. LOSS FUNCTIONS
# ===================================================================

class DiceLoss(nn.Module):
    """Sørensen–Dice coefficient loss for binary masks.

    ``loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)``
    """

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
    """Combined mask (Dice + BCE) and class (CE) loss.

    ``total = λ_mask * (α * Dice + β * BCE) + λ_class * CE``
    """

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_class: float = 1.0,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_class = lambda_class
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        mask_target: torch.Tensor,
        class_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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


# ===================================================================
# 2. COLLATION
# ===================================================================

def collate_fn(batch: List[dict]) -> dict:
    """Collate dataset items into a batched dict with unified labels.

    Ground-truth masks are spatially downsampled from 1024x1024 to 256x256
    to match the mask decoder output resolution.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = torch.stack([item["box_prompt"] for item in batch])

    # Resize masks from (B, 1, 1024, 1024) to (B, 1, 256, 256)
    masks = torch.nn.functional.interpolate(
        masks, size=(256, 256), mode="nearest",
    )

    mapped = [map_dataset_item(item) for item in batch]
    class_labels = torch.tensor(
        [item["class_label"] for item in mapped], dtype=torch.long
    )

    return {
        "image": images,
        "mask": masks,
        "box_prompt": boxes,
        "class_label": class_labels,
        "source_dataset": [item["source_dataset"] for item in batch],
    }


# ===================================================================
# 3. COMBINED DATASET
# ===================================================================

def build_combined_dataset(root_dir: str, split: str = "train") -> ConcatDataset:
    """ConcatDataset of all available datasets for *split*."""
    datasets = []
    for ds_name in SUPPORTED_DATASETS:
        try:
            ds = UniversalIndustrialDataset(root_dir, ds_name, split=split)
            if len(ds) > 0:
                datasets.append(ds)
                logger.info(f"  {ds_name:20s}  {split:5s}  {len(ds):5d} samples")
        except Exception as e:
            logger.warning(f"  {ds_name:20s}  {split:5s}  ERROR: {e}")

    if not datasets:
        raise RuntimeError(f"No datasets available for split='{split}'")
    return ConcatDataset(datasets)


# ===================================================================
# 4. TRAIN / VALIDATE
# ===================================================================

def train_one_epoch(
    model: IndustrialSAM,
    loader: DataLoader,
    criterion: JointMaskClassLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = defaultdict(float)
    num_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        boxes = batch["box_prompt"].to(device)
        class_label = batch["class_label"].to(device)

        optimizer.zero_grad()
        mask_logits, class_logits = model(image, boxes=boxes)
        total, losses = criterion(mask_logits, class_logits, mask, class_label)
        total.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: IndustrialSAM,
    loader: DataLoader,
    criterion: JointMaskClassLoss,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_losses: Dict[str, float] = defaultdict(float)
    num_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        boxes = batch["box_prompt"].to(device)
        class_label = batch["class_label"].to(device)

        mask_logits, class_logits = model(image, boxes=boxes)
        _, losses = criterion(mask_logits, class_logits, mask, class_label)

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


# ===================================================================
# 5. CHECKPOINTING
# ===================================================================

def save_checkpoint(
    model: IndustrialSAM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
):
    ckpt = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_vocab": model.class_vocab,
    }
    torch.save(ckpt, path)
    logger.info(
        "Checkpoint saved: %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss
    )


def load_checkpoint(
    path: str,
    model: IndustrialSAM,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Checkpoint loaded: %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt["val_loss"]


# ===================================================================
# 6. MAIN
# ===================================================================

def _build_optimizer(
    model: IndustrialSAM,
    lr_adapter: float,
    lr_head: float,
    lr_other: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    adapter_params = []
    head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _ADAPTER_PREFIX in name:
            adapter_params.append(param)
        elif "classifier_head" in name:
            head_params.append(param)
        else:
            other_params.append(param)

    logger.info(
        "Parameter groups:  adapters=%d  head=%d  other=%d",
        len(adapter_params), len(head_params), len(other_params),
    )

    return torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": lr_adapter},
            {"params": head_params, "lr": lr_head},
            {"params": other_params, "lr": lr_other},
        ],
        weight_decay=weight_decay,
    )


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────
    logger.info("Loading datasets ...")
    train_dataset = build_combined_dataset(args.root_dir, "train")
    val_dataset = build_combined_dataset(args.root_dir, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────
    model = IndustrialSAM(args.checkpoint, device=device)
    model.to(device)
    logger.info(
        "Model: %d trainable params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ── Optimiser & Loss ──────────────────────────────────────────
    optimizer = _build_optimizer(
        model, args.lr_adapter, args.lr_head, args.lr_other, args.weight_decay,
    )
    criterion = JointMaskClassLoss(
        lambda_mask=args.lambda_mask, lambda_class=args.lambda_class,
    )

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)

    logger.info(
        "Starting training  epochs=%d  batch_size=%d  lr_adapter=%.0e  lr_head=%.0e",
        args.epochs, args.batch_size, args.lr_adapter, args.lr_head,
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.clip_grad,
        )
        val_losses = validate(model, val_loader, criterion, device)

        elapsed = time.perf_counter() - t0

        logger.info(
            "E %03d/%03d  train=%.4f  val=%.4f  "
            "mask=%.4f  cls=%.4f  dice=%.4f  bce=%.4f  %.1fs",
            epoch + 1, args.epochs,
            train_losses["loss"], val_losses["loss"],
            val_losses["mask_loss"], val_losses["class_loss"],
            val_losses["dice_loss"], val_losses["bce_loss"],
            elapsed,
        )

        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            save_checkpoint(model, optimizer, epoch, best_val_loss, args.save_path)

    logger.info("Done.  Best val_loss: %.4f  Checkpoint: %s", best_val_loss, args.save_path)


# ===================================================================
# 7. CLI
# ===================================================================

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train IndustrialSAM on all 6 defect datasets.",
    )
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth",
                        help="Path to SAM ViT-B checkpoint")
    parser.add_argument("--root-dir", default="G:/Dataset",
                        help="Root directory containing all dataset folders")
    parser.add_argument("--save-path", default="best_model.pth",
                        help="Path to save the best checkpoint")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-other", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-class", type=float, default=1.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args())
