"""
Training loop for IndustrialSAM.

Supports:
  - Multi-dataset training (6 industrial defect datasets)
  - AMP mixed precision (mandatory for 6 GB VRAM)
  - Gradient accumulation (default 8 steps)
  - Gradient checkpointing on encoder
  - Cosine annealing LR scheduler
  - Per-dataset loss logging and VRAM tracking
  - Checkpoint save / resume (model + optimizer + scheduler)
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import UniversalIndustrialDataset, SUPPORTED_DATASETS
from label_mapping import map_dataset_item
from losses import DiceLoss, JointMaskClassLoss
from model import IndustrialSAM
from model_setup import _ADAPTER_PREFIX
from optimizer_setup import build_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def collate_fn(batch: List[dict]) -> dict:
    """Collate a list of dataset items into a mini-batch dict.

    Stacks images, masks, and box prompts; maps per-dataset class labels
    to unified 8-class vocabulary via ``map_dataset_item``.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = torch.stack([item["box_prompt"] for item in batch])
    mapped = [map_dataset_item(item) for item in batch]
    class_labels = torch.tensor(
        [item["class_label"] for item in mapped],
        dtype=torch.long,
    )
    return {
        "image": images,
        "mask": masks,
        "box_prompt": boxes,
        "class_label": class_labels,
        "source_dataset": [item["source_dataset"] for item in batch],
    }


def build_combined_dataset(
    root_dir: str,
    split: str = "train",
    max_samples: int = 0,
) -> ConcatDataset:
    """Load all 6 datasets for *split* and concatenate them.

    Returns a ``ConcatDataset`` (or ``Subset`` if *max_samples* > 0).
    """
    datasets = []
    for ds_name in SUPPORTED_DATASETS:
        try:
            ds = UniversalIndustrialDataset(root_dir, ds_name, split=split)
            if len(ds) > 0:
                datasets.append(ds)
                logger.info("  %-20s  %-5s  %5d samples", ds_name, split, len(ds))
        except Exception as e:
            logger.warning("  %-20s  %-5s  ERROR: %s", ds_name, split, e)
    if not datasets:
        raise RuntimeError(f"No datasets available for split='{split}'")
    combined = ConcatDataset(datasets)
    if max_samples > 0 and len(combined) > max_samples:
        logger.info(
            "  Truncating %s to %d samples (of %d)", split, max_samples, len(combined)
        )
        combined = torch.utils.data.Subset(combined, list(range(max_samples)))
    return combined


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    clip_grad=1.0,
    grad_accum=1,
    scaler=None,
):
    """Run one training epoch.

    Returns
    -------
    avg_losses : dict  Aggregate average losses {"loss", "mask_loss", ...}.
    avg_dataset : dict  Per-dataset average losses keyed by source name.
    """
    model.train()
    # Keep encoder in eval mode (frozen — no dropout / batchnorm updates)
    model.image_encoder.eval()

    total_losses = defaultdict(float)
    dataset_losses = defaultdict(lambda: defaultdict(float))
    dataset_counts = defaultdict(int)
    num_batches = 0
    optimizer.zero_grad()
    t_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box_prompt"].to(device, non_blocking=True)
        class_label = batch["class_label"].to(device, non_blocking=True)
        sources = batch["source_dataset"]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            mask_logits, class_logits = model(image, boxes=boxes)
            total, losses = criterion(
                mask_logits.float(),
                class_logits.float(),
                mask.float(),
                class_label,
            )

        if scaler is not None:
            scaler.scale(total).backward()
        else:
            total.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if scaler is not None:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad()

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

        # Per-dataset accumulation (use batch size as weight)
        bs = len(sources)
        for i, src in enumerate(sources):
            dataset_losses[src]["loss"] += losses["loss"].item()
            dataset_losses[src]["mask_loss"] += losses["mask_loss"].item()
            dataset_losses[src]["class_loss"] += losses["class_loss"].item()
            dataset_counts[src] += 1

        if (batch_idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            avg = total.item() / (batch_idx + 1)
            logger.info(
                "  train batch %d/%d  loss=%.4f  %.1fs",
                batch_idx + 1,
                len(loader),
                avg,
                elapsed,
            )

    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    avg_dataset = {}
    for src in sorted(dataset_losses):
        cnt = dataset_counts[src]
        avg_dataset[src] = {k: v / cnt for k, v in dataset_losses[src].items()}
    return avg_losses, avg_dataset


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation (no gradient computation).

    Returns
    -------
    avg_losses : dict  Aggregate average losses.
    avg_dataset : dict  Per-dataset average losses keyed by source name.
    """
    model.eval()
    total_losses = defaultdict(float)
    dataset_losses = defaultdict(lambda: defaultdict(float))
    dataset_counts = defaultdict(int)
    num_batches = 0
    t_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box_prompt"].to(device, non_blocking=True)
        class_label = batch["class_label"].to(device, non_blocking=True)
        sources = batch["source_dataset"]

        mask_logits, class_logits = model(image, boxes=boxes)
        _, losses = criterion(mask_logits, class_logits, mask, class_label)

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

        for i, src in enumerate(sources):
            dataset_losses[src]["loss"] += losses["loss"].item()
            dataset_losses[src]["mask_loss"] += losses["mask_loss"].item()
            dataset_losses[src]["class_loss"] += losses["class_loss"].item()
            dataset_counts[src] += 1

        if (batch_idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            avg = total_losses["loss"] / num_batches
            logger.info(
                "  val   batch %d/%d  loss=%.4f  %.1fs",
                batch_idx + 1,
                len(loader),
                avg,
                elapsed,
            )

    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    avg_dataset = {}
    for src in sorted(dataset_losses):
        cnt = dataset_counts[src]
        avg_dataset[src] = {k: v / cnt for k, v in dataset_losses[src].items()}
    return avg_losses, avg_dataset


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    """Save model + optimizer + scheduler state to *path*."""
    ckpt = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_vocab": model.class_vocab,
    }
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Restore model (and optionally optimizer + scheduler) from *path*.

    Returns
    -------
    start_epoch : int  Epoch to resume from.
    val_loss : float   Validation loss at save time.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Checkpoint loaded: %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt["val_loss"]


def main(args):
    """Entry point: build datasets, model, optimizer, and run training loop."""
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            props.total_memory / 1024**3,
        )

    logger.info("Loading datasets ...")
    train_dataset = build_combined_dataset(args.root_dir, "train", args.max_samples)
    val_dataset = build_combined_dataset(args.root_dir, "val", args.max_samples)

    # Precompute cache — eliminates disk I/O during training
    # Skip when max_samples is set (Subset only needs a few items, not full cache)
    if args.max_samples == 0:
        logger.info("Precomputing cache ...")
        underlying = (
            train_dataset.dataset
            if hasattr(train_dataset, "dataset")
            else train_dataset
        )
        for ds in underlying.datasets:
            try:
                ds.precompute_cache()
            except MemoryError:
                logger.warning(
                    "Not enough RAM to cache %s — skipping cache", ds.dataset_name
                )
        underlying_val = (
            val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset
        )
        for ds in underlying_val.datasets:
            try:
                ds.precompute_cache()
            except MemoryError:
                logger.warning(
                    "Not enough RAM to cache %s — skipping cache", ds.dataset_name
                )
        logger.info("Cache ready.")

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    model = IndustrialSAM(
        args.checkpoint,
        device=device,
        train_resolution=args.train_resolution,
    )
    logger.info(
        "Model: %d trainable params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = build_optimizer(
        model, lr=args.lr_adapter, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_adapter * 0.01
    )
    criterion = JointMaskClassLoss(
        lambda_mask=args.lambda_mask,
        lambda_class=args.lambda_class,
    ).to(device)

    scaler = None
    if args.amp and device == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info(
            "AMP: ON  |  grad_accum: %d  |  effective batch: %d",
            args.grad_accum,
            args.batch_size * args.grad_accum,
        )
    else:
        logger.info("AMP: OFF  |  grad_accum: %d", args.grad_accum)

    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)

    logger.info(
        "Training: epochs=%d  batch=%d  grad_accum=%d  lr_adapter=%.0e  lr_head=%.0e  resolution=%d",
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.lr_adapter,
        args.lr_head,
        args.train_resolution,
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_losses, train_ds = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.clip_grad,
            args.grad_accum,
            scaler,
        )
        val_losses, val_ds = validate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0

        logger.info(
            "E %03d/%03d  lr=%.2e  train=%.4f  val=%.4f  mask=%.4f  cls=%.4f  dice=%.4f  bce=%.4f  %.1fs",
            epoch + 1,
            args.epochs,
            current_lr,
            train_losses["loss"],
            val_losses["loss"],
            val_losses["mask_loss"],
            val_losses["class_loss"],
            val_losses["dice_loss"],
            val_losses["bce_loss"],
            elapsed,
        )

        # Per-dataset breakdown
        if val_ds:
            parts = []
            for src in sorted(val_ds):
                parts.append(f"{src}={val_ds[src]['loss']:.4f}")
            logger.info("  val  per-dataset: %s", "  ".join(parts))
        if train_ds:
            parts = []
            for src in sorted(train_ds):
                parts.append(f"{src}={train_ds[src]['loss']:.4f}")
            logger.info("  trn  per-dataset: %s", "  ".join(parts))

        # VRAM usage
        if device == "cuda":
            cur = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            logger.info("  vram  cur=%.0fMB  peak=%.0fMB", cur, peak)

        scheduler.step()

        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, args.save_path
            )

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logger.info(
        "Done.  Best val_loss: %.4f  Checkpoint: %s",
        best_val_loss,
        args.save_path,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train IndustrialSAM on all 6 defect datasets.",
    )
    parser.add_argument(
        "--checkpoint",
        default="sam_vit_b_01ec64.pth",
        help="Path to SAM ViT-B checkpoint",
    )
    parser.add_argument(
        "--root-dir",
        default="D:/Dataset",
        help="Root directory containing all dataset folders",
    )
    parser.add_argument(
        "--save-path",
        default="best_model.pth",
        help="Path to save the best checkpoint",
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size (6 GB VRAM -> 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (8-16 for 6 GB)",
    )
    parser.add_argument(
        "--train-resolution",
        type=int,
        default=512,
        help="Training resolution (512 recommended for 6 GB VRAM)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable AMP mixed precision (mandatory for 6 GB)",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-other", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-class", type=float, default=0.5)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (4-6 recommended)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap dataset size (0 = use all)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args())
