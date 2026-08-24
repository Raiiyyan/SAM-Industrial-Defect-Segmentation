"""
evaluate.py — Statistical evaluation and metric matrix for IndustrialSAM.

Produces:
  results/metrics.csv          — structured metric table
  results/metrics_table.md     — markdown table for reports
  results/confusion_matrix.png — 8x8 heatmap
  results/per_class_report.csv — per-class precision / recall / F1
  results/failure_cases/       — worst 20 predictions as side-by-side PNGs
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

from dataset import UniversalIndustrialDataset, SUPPORTED_DATASETS, TARGET_SIZE
from label_mapping import map_dataset_item
from classifier_head import CLASS_VOCAB
from model import IndustrialSAM
from model_setup import _ADAPTER_PREFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def collate_fn(batch):
    """Collate dataset items into a batch dict for evaluation.

    Includes ``image_path`` and ``meta`` for failure-case reporting.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = torch.stack([item["box_prompt"] for item in batch])
    mapped = [map_dataset_item(item) for item in batch]
    class_labels = torch.tensor([m["class_label"] for m in mapped], dtype=torch.long)
    return {
        "image": images,
        "mask": masks,
        "box_prompt": boxes,
        "class_label": class_labels,
        "source_dataset": [item["source_dataset"] for item in batch],
        "image_path": [item.get("image_path", "") for item in batch],
        "meta": [item.get("meta", {}) for item in batch],
    }


def build_test_dataset(root_dir, max_samples=0):
    """Load all 6 test splits and concatenate into a single dataset."""
    datasets = []
    for ds_name in SUPPORTED_DATASETS:
        try:
            ds = UniversalIndustrialDataset(root_dir, ds_name, split="test")
            if len(ds) > 0:
                datasets.append(ds)
                logger.info("  %-20s test %5d samples", ds_name, len(ds))
        except Exception as e:
            logger.warning("  %-20s test ERROR: %s", ds_name, e)
    if not datasets:
        raise RuntimeError("No test datasets found")
    from torch.utils.data import ConcatDataset

    combined = ConcatDataset(datasets)
    if max_samples > 0 and len(combined) > max_samples:
        combined = torch.utils.data.Subset(combined, list(range(max_samples)))
    return combined


@torch.no_grad()
def run_evaluation(model, loader, device, num_classes=8):
    """Run inference on the test set and collect per-sample metrics.

    Returns
    -------
    dict with keys: mask_ious, mask_dices, preds, labels, sources,
    image_paths, confidences, failure_cases.
    """
    model.eval()
    all_mask_ious = []
    all_mask_dices = []
    all_preds = []
    all_labels = []
    all_sources = []
    all_image_paths = []
    all_confidences = []
    failure_cases = []

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        mask_gt = batch["mask"].to(device, non_blocking=True)
        boxes = batch["box_prompt"].to(device, non_blocking=True)
        class_label = batch["class_label"]

        with torch.amp.autocast("cuda"):
            mask_logits, class_logits = model(image, boxes=boxes)

        # Resize GT mask to match prediction size
        if mask_gt.shape[2:] != mask_logits.shape[2:]:
            mask_gt = F.interpolate(
                mask_gt,
                size=mask_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Segmentation metrics
        mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()
        for i in range(mask_pred.shape[0]):
            pred_flat = mask_pred[i].flatten()
            gt_flat = mask_gt[i].flatten()
            intersection = (pred_flat * gt_flat).sum().item()
            union = pred_flat.sum().item() + gt_flat.sum().item() - intersection
            iou = intersection / max(union, 1e-6)
            dice = (
                2
                * intersection
                / max(pred_flat.sum().item() + gt_flat.sum().item(), 1e-6)
            )
            all_mask_ious.append(iou)
            all_mask_dices.append(dice)

        # Classification metrics
        probs = F.softmax(class_logits, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(class_label.tolist())
        all_sources.extend(batch["source_dataset"])
        all_confidences.extend(confidences.cpu().tolist())
        all_image_paths.extend(batch["image_path"])

        # Collect worst cases for failure analysis
        for i in range(mask_pred.shape[0]):
            failure_cases.append(
                {
                    "iou": all_mask_ious[-(mask_pred.shape[0] - i)],
                    "pred": preds[i].item(),
                    "label": class_label[i].item(),
                    "confidence": confidences[i].item(),
                    "source": batch["source_dataset"][i],
                    "image_path": batch["image_path"][i],
                    "mask_gt": mask_gt[i].cpu(),
                    "mask_pred": mask_pred[i].cpu(),
                    "image": batch["image"][i].cpu(),
                }
            )

        if (batch_idx + 1) % 50 == 0:
            logger.info("  eval batch %d/%d", batch_idx + 1, len(loader))

    return {
        "mask_ious": all_mask_ious,
        "mask_dices": all_mask_dices,
        "preds": all_preds,
        "labels": all_labels,
        "sources": all_sources,
        "image_paths": all_image_paths,
        "confidences": all_confidences,
        "failure_cases": failure_cases,
    }


def compute_metrics(results, num_classes=8, class_names=None):
    """Aggregate raw results into structured metrics.

    Returns
    -------
    dict with keys: miou, dice, per_dataset, confusion_matrix,
    per_class, overall_accuracy, macro_f1.
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Overall segmentation
    miou = np.mean(results["mask_ious"])
    dice = np.mean(results["mask_dices"])

    # Per-dataset segmentation
    dataset_ious = defaultdict(list)
    dataset_dices = defaultdict(list)
    for i, src in enumerate(results["sources"]):
        dataset_ious[src].append(results["mask_ious"][i])
        dataset_dices[src].append(results["mask_dices"][i])

    per_dataset = {}
    for src in sorted(dataset_ious.keys()):
        per_dataset[src] = {
            "miou": np.mean(dataset_ious[src]),
            "dice": np.mean(dataset_dices[src]),
            "n_samples": len(dataset_ious[src]),
        }

    # Classification
    labels = np.array(results["labels"])
    preds = np.array(results["preds"])

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    overall_acc = accuracy_score(labels, preds)
    macro_f1 = np.mean(f1)

    per_class = []
    for i in range(num_classes):
        per_class.append(
            {
                "class": i,
                "name": class_names[i],
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
        )

    return {
        "miou": miou,
        "dice": dice,
        "per_dataset": per_dataset,
        "confusion_matrix": cm,
        "per_class": per_class,
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
    }


def save_confusion_matrix(cm, class_names, output_path):
    """Render and save an 8x8 confusion matrix heatmap as PNG."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", output_path)


def save_failure_cases(failure_cases, output_dir, n=20):
    """Save the *n* worst predictions as side-by-side PNG visualisations."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort by lowest IoU (worst segmentation) or misclassification
    def score(fc):
        seg_score = fc["iou"]
        cls_score = 1.0 if fc["pred"] != fc["label"] else 0.0
        return -(seg_score + cls_score * 0.5)

    failure_cases.sort(key=score)
    worst = failure_cases[:n]

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, fc in enumerate(worst):
        img = fc["image"] * IMAGENET_STD + IMAGENET_MEAN
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        gt = fc["mask_gt"][0].numpy()
        pred = fc["mask_pred"][0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(pred, cmap="Reds", alpha=0.4)
        correct = fc["pred"] == fc["label"]
        axes[2].set_title(
            f"Pred: {fc['pred']} ({fc['confidence']:.1%}) "
            f"{'OK' if correct else 'WRONG'}"
        )
        axes[2].axis("off")

        fig.suptitle(f"IoU={fc['iou']:.3f} | {fc['source']}", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"failure_{idx:03d}.png"), dpi=100)
        plt.close(fig)

    logger.info("Saved %d failure cases to %s", len(worst), output_dir)


def save_metric_charts(metrics, results, output_dir):
    """Generate and save all evaluation metric charts."""
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Per-Class F1 Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    class_names = [pc["name"] for pc in metrics["per_class"]]
    f1_scores = [pc["f1"] for pc in metrics["per_class"]]
    colors = plt.cm.RdYlGn([f for f in f1_scores])
    bars = ax.barh(class_names, f1_scores, color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_xlim(0, 1.0)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.3f}", va="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-Class Precision/Recall/F1 Grouped Bar Chart ──
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(class_names))
    width = 0.25
    precision = [pc["precision"] for pc in metrics["per_class"]]
    recall = [pc["recall"] for pc in metrics["per_class"]]
    ax.bar([i - width for i in x], precision, width, label="Precision", color="#2196F3")
    ax.bar(x, recall, width, label="Recall", color="#FF9800")
    ax.bar([i + width for i in x], f1_scores, width, label="F1", color="#4CAF50")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_class_precision_recall_f1.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-Dataset mIoU & Dice Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = list(metrics["per_dataset"].keys())
    mious = [metrics["per_dataset"][d]["miou"] for d in datasets]
    dices = [metrics["per_dataset"][d]["dice"] for d in datasets]
    x = range(len(datasets))
    width = 0.35
    ax.bar([i - width/2 for i in x], mious, width, label="mIoU", color="#9C27B0")
    ax.bar([i + width/2 for i in x], dices, width, label="Dice", color="#E91E63")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.set_title("Per-Dataset Segmentation Performance (mIoU & Dice)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_dataset_miou_dice.png"), dpi=150)
    plt.close(fig)

    # ── 4. IoU Distribution Histogram ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ious = results["mask_ious"]
    ax.hist(ious, bins=50, color="#3F51B5", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(ious), color="red", linestyle="--", linewidth=2, label=f"Mean IoU: {np.mean(ious):.4f}")
    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mask IoU Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iou_distribution.png"), dpi=150)
    plt.close(fig)

    # ── 5. Confidence Distribution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    confs = results["confidences"]
    ax.hist(confs, bins=50, color="#009688", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(confs), color="red", linestyle="--", linewidth=2, label=f"Mean Confidence: {np.mean(confs):.4f}")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Classification Confidence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close(fig)

    # ── 6. Summary Dashboard ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("IndustrialSAM Evaluation Dashboard", fontsize=16, fontweight="bold")

    # (0,0) Overall metrics bar
    ax = axes[0, 0]
    overall = {"mIoU": metrics["miou"], "Dice": metrics["dice"], "Accuracy": metrics["overall_accuracy"], "Macro F1": metrics["macro_f1"]}
    bars = ax.bar(overall.keys(), overall.values(), color=["#9C27B0", "#E91E63", "#2196F3", "#4CAF50"])
    for bar, val in zip(bars, overall.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall Metrics")
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Per-class F1
    ax = axes[0, 1]
    ax.barh(class_names, f1_scores, color=plt.cm.RdYlGn(f1_scores))
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-Class F1")
    ax.grid(True, alpha=0.3, axis="x")

    # (0,2) Per-dataset Dice
    ax = axes[0, 2]
    ax.barh(datasets, dices, color="#E91E63")
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-Dataset Dice")
    ax.grid(True, alpha=0.3, axis="x")

    # (1,0) IoU distribution
    ax = axes[1, 0]
    ax.hist(ious, bins=40, color="#3F51B5", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(ious), color="red", linestyle="--", linewidth=2)
    ax.set_title(f"IoU Distribution (mean={np.mean(ious):.4f})")
    ax.grid(True, alpha=0.3)

    # (1,1) Confusion matrix (already saved separately, show summary)
    ax = axes[1, 1]
    cm = metrics["confusion_matrix"]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=6)
    plt.colorbar(im, ax=ax)

    # (1,2) Support per class
    ax = axes[1, 2]
    supports = [pc["support"] for pc in metrics["per_class"]]
    ax.barh(class_names, supports, color="#FF5722")
    ax.set_title("Samples per Class (Support)")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "evaluation_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved evaluation charts to %s", output_dir)


def save_results(metrics, output_dir):
    """Write metrics.csv, metrics_table.md, and per_class_report.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # metrics.csv
    rows = []
    rows.append(["Overall mIoU", f"{metrics['miou']:.4f}"])
    rows.append(["Overall Dice", f"{metrics['dice']:.4f}"])
    rows.append(["Overall Accuracy", f"{metrics['overall_accuracy']:.4f}"])
    rows.append(["Macro F1", f"{metrics['macro_f1']:.4f}"])
    rows.append([])
    for src, m in metrics["per_dataset"].items():
        rows.append([f"{src}/mIoU", f"{m['miou']:.4f}"])
        rows.append([f"{src}/Dice", f"{m['dice']:.4f}"])
        rows.append([f"{src}/n", str(m["n_samples"])])

    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    logger.info("Metrics CSV saved: %s", os.path.join(output_dir, "metrics.csv"))

    # metrics_table.md
    lines = ["# Evaluation Results\n"]
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| mIoU | {metrics['miou']:.4f} |")
    lines.append(f"| Dice | {metrics['dice']:.4f} |")
    lines.append(f"| Accuracy | {metrics['overall_accuracy']:.4f} |")
    lines.append(f"| Macro F1 | {metrics['macro_f1']:.4f} |")

    lines.append("\n## Per-Dataset Segmentation\n")
    lines.append("| Dataset | mIoU | Dice | Samples |")
    lines.append("|---------|------|------|---------|")
    for src, m in metrics["per_dataset"].items():
        lines.append(
            f"| {src} | {m['miou']:.4f} | {m['dice']:.4f} | {m['n_samples']} |"
        )

    lines.append("\n## Per-Class Classification\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|-----|---------|")
    for pc in metrics["per_class"]:
        lines.append(
            f"| {pc['name']} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )

    with open(os.path.join(output_dir, "metrics_table.md"), "w") as f:
        f.write("\n".join(lines))
    logger.info("Metrics table saved: %s", os.path.join(output_dir, "metrics_table.md"))

    # per_class_report.csv
    with open(os.path.join(output_dir, "per_class_report.csv"), "w") as f:
        f.write("class,name,precision,recall,f1,support\n")
        for pc in metrics["per_class"]:
            f.write(
                f"{pc['class']},{pc['name']},{pc['precision']:.4f},"
                f"{pc['recall']:.4f},{pc['f1']:.4f},{pc['support']}\n"
            )
    logger.info(
        "Per-class report saved: %s", os.path.join(output_dir, "per_class_report.csv")
    )


CLASS_NAMES = [CLASS_VOCAB[i] for i in range(len(CLASS_VOCAB))]


def main(args):
    """Entry point: load model, run evaluation, save all result artefacts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading test dataset ...")
    test_dataset = build_test_dataset(args.root_dir, args.max_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    logger.info("Test samples: %d", len(test_dataset))

    logger.info("Loading model from %s ...", args.checkpoint)
    model = IndustrialSAM(args.sam_checkpoint, device=device, train_resolution=512)
    # Load trained weights (adapters + classifier head) from training checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info(
        "Model loaded. Trainable params: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    logger.info("Running evaluation ...")
    t0 = time.perf_counter()
    results = run_evaluation(model, test_loader, device)
    elapsed = time.perf_counter() - t0
    logger.info("Evaluation complete in %.1fs", elapsed)

    logger.info("Computing metrics ...")
    metrics = compute_metrics(results, num_classes=8, class_names=CLASS_NAMES)

    # Log summary
    logger.info("=== RESULTS ===")
    logger.info("  mIoU:            %.4f", metrics["miou"])
    logger.info("  Dice:            %.4f", metrics["dice"])
    logger.info("  Accuracy:        %.4f", metrics["overall_accuracy"])
    logger.info("  Macro F1:        %.4f", metrics["macro_f1"])
    for src, m in metrics["per_dataset"].items():
        logger.info(
            "  %s: IoU=%.4f  Dice=%.4f  (n=%d)",
            src,
            m["miou"],
            m["dice"],
            m["n_samples"],
        )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Saving results ...")
    save_results(metrics, args.output_dir)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    save_failure_cases(
        results["failure_cases"],
        os.path.join(args.output_dir, "failure_cases"),
        n=args.n_failures,
    )
    save_metric_charts(metrics, results, args.output_dir)

    logger.info("Done.  All results in %s", args.output_dir)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate IndustrialSAM on test sets.")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--checkpoint", default=os.path.join(_script_dir, "checkpoints", "best_model.pth"),
                        help="Training checkpoint with trained weights")
    parser.add_argument("--sam-checkpoint", default=os.path.join(_script_dir, "sam_vit_b_01ec64.pth"),
                        help="Base SAM ViT-B checkpoint")
    parser.add_argument("--root-dir", default=r"G:\Dataset")
    parser.add_argument("--output-dir", default=os.path.join(_script_dir, "results"))
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Cap test set size (0 = all)"
    )
    parser.add_argument(
        "--n-failures", type=int, default=20, help="Number of failure cases to save"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args())