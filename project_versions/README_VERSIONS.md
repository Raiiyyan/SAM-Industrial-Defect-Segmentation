# Project Evolution — Three Versions

## Version 1 — `v1_demo_jul27.py` (Jul 12/27, 2026)
**File in repo:** `demo.py` (deleted later)
- **Purpose:** One-off script to prove SAM works on MVTec data.
- **How it works:**
  - Hardcoded image: `bottle/test/broken_large/000.png`
  - Loads base SAM (ViT-B) with `SamPredictor`
  - Uses a **fixed box prompt** covering 10%–90% of the image
  - Runs once, shows result in a **matplotlib window** (no web UI)
  - Simulates a class label: "Broken / Structural Crack (Class 1)"
- **Key limitation:** No automation, no YOLO, fixed image, fixed prompt.

---

## Version 2 — `v2_sam_finetune/` (Jul 28, 2026 — commit ff46b99)
**Files in repo:** `train.py`, `evaluate.py`, `demo_live.py`, `model.py`, `dataset.py`, `adapters.py`, `classifier_head.py`, `label_mapping.py`, `losses.py`, `model_setup.py`, `optimizer_setup.py`, `requirements.txt`, `README.md`
- **Purpose:** Full **SAM fine-tuning** pipeline — train SAM itself (not YOLO) on 6 industrial datasets.
- **Key difference from v3:** Fine-tunes SAM directly (with adapters + classifier head), NOT YOLO.
- **How it works:**
  1. `dataset.py` — loads 6 datasets (MVTec AD, MVTec AD 2, Severstal, NEU-DET, DAGM2007, DefectSpectrum) with letterboxing to 512×512.
  2. `adapters.py` — lightweight bottleneck adapters (768→64→768) injected into frozen SAM encoder.
  3. `model.py` — `IndustrialSAM` wrapper: frozen encoder + adapters + mask decoder + classifier head.
  4. `train.py` — trains with AMP, gradient accumulation, cosine LR, checkpoint save/resume.
  5. `evaluate.py` — runs test set, computes mIoU/Dice/F1, confusion matrix, failure cases.
  6. `demo_live.py` — live faculty demo: datasets, label mapping, model architecture, inference, training pipeline, unit tests.
- **Architecture:** SAM encoder (frozen) → 24 adapters → mask decoder → mask + 8-class classification.
- **8-class unified vocabulary:** Flawless, Surface Scratch, Structural Crack, Hole/Puncture, Inclusion, Missing Component, Discoloration/Stain, Geometric Deformation.
- **93.26% frozen; 6.5M trainable params** (adapters + classifier head).
- **This was the "v2" you remembered.** Deleted from repo on Aug 16 (commit 6a7f036 "Delete Sam_seg.py").

---

## Version 3 — `v3_app_aug16.py` (Aug 16, 2026 → current `app.py`)
**File in repo:** `app.py`, `train_yolo.py`, `generate_data.py`, `data.yaml`
- **Purpose:** Full **hybrid inspection pipeline** (YOLO + SAM).
- **How it works:**
  1. **YOLOv8** (fine-tuned on defect data via `train_yolo.py`) detects defects → bounding boxes.
  2. **SAM** (`SamPredictor`, not auto-generator) takes each YOLO box as a **prompt** → precise pixel mask.
  3. Red overlay = defect pixels; Yellow box = YOLO detection.
  4. PASS/FAIL verdict + total defect pixel count.
  5. Confidence threshold slider to tune sensitivity.
- **Training data pipeline:** `generate_data.py` converts MVTec masks → YOLO labels → 80/20 split → `train_yolo.py` fine-tunes `yolov8m.pt` → `best.pt` used here.

---

## Evolution Summary

| Version | Date | Approach | Model(s) | Prompt | UI | Trains? |
|---|---|---|---|---|---|---|
| v1 | Jul 12/27 | Prove SAM works | SAM only | Fixed box (10–90%) | Matplotlib | No |
| v2 | Jul 28 | Fine-tune SAM | SAM + adapters + classifier | Box from mask | `demo_live.py` console | **Yes (adapters + head)** |
| v3 | Aug 16+ | YOLO+SAM hybrid | YOLO (trained) + SAM (frozen) | YOLO box → SAM | Gradio web UI | **Yes (YOLO)** |

**Key architectural difference:**
- **v2** fine-tunes SAM itself (adds adapters inside SAM's encoder, adds a classifier head, jointly predicts mask + defect class). SAM learns to both segment AND classify defects.
- **v3** keeps SAM frozen and trains YOLO separately. YOLO handles detection, SAM only does segmentation prompted by YOLO boxes.