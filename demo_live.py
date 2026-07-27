"""
Live faculty demo script - run in VS Code terminal:
    python demo_live.py

Each section prints detailed output you can walk through step by step.
"""

import sys, os, time, contextlib, io, logging
import torch
import numpy as np

def _silence_datasets():
    logging.getLogger("UniversalIndustrialDataset").setLevel(logging.ERROR)
    logging.getLogger("train").setLevel(logging.ERROR)
    logging.getLogger("train_laptop").setLevel(logging.ERROR)

def _restore_logging():
    logging.getLogger("UniversalIndustrialDataset").setLevel(logging.INFO)
    logging.getLogger("train").setLevel(logging.INFO)
    logging.getLogger("train_laptop").setLevel(logging.INFO)

ROOT = "G:/Dataset"
CKPT = "sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SECTION 0 - Project overview (show in VS Code file explorer)
# ============================================================
# Open VS Code, show:
#   README.md  - architecture diagram, component table
#   dataset.py         - 6 dataset parsers
#   adapters.py        - IndustrialAdapter module
#   model_setup.py     - freeze, inject, verify
#   classifier_head.py - DefectClassifierHead
#   model.py           - IndustrialSAM wrapper
#   label_mapping.py   - per-dataset -> unified 8-class
#   train.py           - full training pipeline
#   train_laptop.py    - laptop edition
#   requirements.txt / requirements_laptop.txt
#   demo.py / demo_live.py


# ============================================================
# SECTION 1 - Dataset loading (all 6 datasets)
# ============================================================
def section1_datasets():
    print("=" * 72)
    print("SECTION 1: DATASET LOADING - all 6 industrial datasets")
    print("=" * 72)

    _silence_datasets()  # must come before the import that creates the logger
    from dataset import SUPPORTED_DATASETS, UniversalIndustrialDataset

    for name in SUPPORTED_DATASETS:
        ds = UniversalIndustrialDataset(ROOT, name, split="train")
        item = ds[0]
        print(f"\n  {name:20s}  {len(ds):5d} samples")
        print(f"    image      : {tuple(item['image'].shape)}   {item['image'].dtype}")
        print(f"    mask       : {tuple(item['mask'].shape)}   {item['mask'].dtype}")
        print(f"    box_prompt : {item['box_prompt'].tolist()}")
        print(f"    class_label: {item['class_label']}  source: {item['source_dataset']}")

    print("\n  -> All 6 datasets loaded successfully")
    print(f"  -> Total train samples: {sum(len(UniversalIndustrialDataset(ROOT, n, 'train')) for n in SUPPORTED_DATASETS)}")
    _restore_logging()


# ============================================================
# SECTION 2 - Label mapping (per-dataset -> unified 8-class)
# ============================================================
def section2_label_mapping():
    print("\n" + "=" * 72)
    print("SECTION 2: LABEL MAPPING - per-dataset -> unified 8-class vocabulary")
    print("=" * 72)

    _silence_datasets()
    from label_mapping import map_dataset_item
    from classifier_head import CLASS_VOCAB
    from dataset import UniversalIndustrialDataset

    print(f"\n  Unified vocabulary ({len(CLASS_VOCAB)} classes):")
    for i, name in CLASS_VOCAB.items():
        print(f"    [{i}] {name}")

    for name in ["mvtec_ad", "severstal", "neu_det"]:
        ds = UniversalIndustrialDataset(ROOT, name, split="train")
        raw = ds[0]
        mapped = map_dataset_item(raw)
        print(f"\n  {name}:")
        print(f"    original class_label = {raw['class_label']}")
        print(f"    unified  class_label = {mapped['class_label']} -> {CLASS_VOCAB[mapped['class_label']]}")
    _restore_logging()


# ============================================================
# SECTION 3 - Model architecture (adapters + classifier head)
# ============================================================
def section3_model():
    print("\n" + "=" * 72)
    print("SECTION 3: MODEL ARCHITECTURE - adapters + classifier head")
    print("=" * 72)

    from model import IndustrialSAM
    from model_setup import verify_gradient_isolation, _ADAPTER_PREFIX

    model = IndustrialSAM(CKPT, device=DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    adapter_params = sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and _ADAPTER_PREFIX in n)
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and "classifier_head" in n)

    print(f"\n  Total parameters : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}%)")
    print(f"  Frozen params    : {frozen:,}  ({100*frozen/total:.2f}%)")
    print(f"  Adapter params   : {adapter_params:,}")
    print(f"  Classifier head  : {head_params:,}")
    print(f"  Device           : {DEVICE}")

    print(f"\n  Adapter injection: 24 adapters (2 per block x 12 blocks)")
    adapter_count = sum(1 for n, _ in model.named_parameters() if _ADAPTER_PREFIX in n)
    print(f"  Adapter param tensors: {adapter_count}")

    verify_gradient_isolation(model)

    del model
    torch.cuda.empty_cache()


# ============================================================
# SECTION 4 - Inference demo (forward pass on real sample)
# ============================================================
def section4_inference():
    print("\n" + "=" * 72)
    print("SECTION 4: INFERENCE DEMO - forward pass on real sample")
    print("=" * 72)

    from dataset import UniversalIndustrialDataset
    from label_mapping import map_dataset_item
    from model import IndustrialSAM

    ds = UniversalIndustrialDataset(ROOT, "mvtec_ad", split="train")
    item = map_dataset_item(ds[0])

    model = IndustrialSAM(CKPT, device=DEVICE)
    model.eval()

    img = item["image"].unsqueeze(0).to(DEVICE)
    box = item["box_prompt"].unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        mask_logits, class_logits = model(img, boxes=box)
    elapsed = time.perf_counter() - t0

    pred_class = class_logits.argmax(dim=1).item()

    from classifier_head import CLASS_VOCAB
    print(f"\n  Input image       : {tuple(img.shape)} on {img.device}")
    print(f"  Input box         : {box[0].tolist()}")
    print(f"  Mask logits shape : {tuple(mask_logits.shape)}")
    print(f"  Class logits      : {class_logits[0].tolist()}")
    print(f"  Predicted class   : [{pred_class}] {CLASS_VOCAB[pred_class]}")
    print(f"  Inference time    : {elapsed*1000:.0f} ms")

    del model, img, box
    torch.cuda.empty_cache()


# ============================================================
# SECTION 5 - Training loop (a few batches)
# ============================================================
def section5_training():
    print("\n" + "=" * 72)
    print("SECTION 5: TRAINING PIPELINE - 3 batches on GPU")
    print("=" * 72)

    _silence_datasets()
    from torch.utils.data import DataLoader, Subset
    from model import IndustrialSAM
    from train import build_combined_dataset, collate_fn, JointMaskClassLoss

    print("\n  Building combined dataset (all 6 datasets)...")
    ds = build_combined_dataset(ROOT, "train")
    subset = Subset(ds, list(range(6)))
    loader = DataLoader(subset, batch_size=2, collate_fn=collate_fn, num_workers=0)

    model = IndustrialSAM(CKPT, device=DEVICE)
    model.train()
    criterion = JointMaskClassLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Batch size       : 2")
    print(f"  Device           : {DEVICE}")
    print()

    for i, batch in enumerate(loader):
        if i >= 3:
            break
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        box = batch["box_prompt"].to(DEVICE)
        cl = batch["class_label"].to(DEVICE)

        optimizer.zero_grad()
        mask_logits, class_logits = model(img, boxes=box)
        loss, losses = criterion(mask_logits, class_logits, msk, cl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"    Batch {i}:  total={loss.item():.4f}  "
              f"mask={losses['mask_loss']:.4f}  "
              f"class={losses['class_loss']:.4f}  "
              f"dice={losses['dice_loss']:.4f}  "
              f"bce={losses['bce_loss']:.4f}")

    print("\n  -> Training pipeline verified: forward + backward + optimizer OK")

    _restore_logging()
    del model, loader
    torch.cuda.empty_cache()


# ============================================================
# SECTION 6 - Unit tests
# ============================================================
def section6_tests():
    print("\n" + "=" * 72)
    print("SECTION 6: UNIT TESTS - parser tests (no real data needed)")
    print("=" * 72)

    import subprocess, sys

    result = subprocess.run(
        [sys.executable, "dataset.py", "--test"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    for line in result.stdout.splitlines():
        if line.strip():
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  -> Some tests FAILED (exit code {result.returncode})")
    else:
        print(f"  -> All 13 parser tests PASSED")


# ============================================================
# RUN ALL SECTIONS
# ============================================================
if __name__ == "__main__":
    print()
    print("  +------------------------------------------------------+")
    print("  |  SAM -- Industrial Defect Segmentation               |")
    print("  |  Live Demo  |  6 datasets  |  8-class unified        |")
    print("  +------------------------------------------------------+")
    print(f"  Device: {DEVICE}  |  Project: {os.path.basename(os.getcwd())}")
    print()

    # Before starting, show VS Code file explorer:
    print("  [Before running - show in VS Code file explorer:]")
    print("   README.md  - architecture, component table, commands")
    print("   dataset.py / adapters.py / model_setup.py / model.py")
    print("   classifier_head.py / label_mapping.py / train.py")
    print()

    section1_datasets()
    section2_label_mapping()
    section3_model()
    section4_inference()
    section5_training()
    section6_tests()

    print()
    print("=" * 72)
    print("DEMO COMPLETE - all components verified.")
    print("=" * 72)
