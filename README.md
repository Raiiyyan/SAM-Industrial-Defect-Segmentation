# SAM Adaptation for Multi-Class Industrial Defect Segmentation

Parameter-efficient fine-tuning of Meta's Segment Anything Model (SAM ViT-B) for joint defect segmentation and classification across 6 industrial datasets.

## Datasets

Download and place each dataset under a common root directory (e.g. `G:/Dataset`). The folder names must match exactly:

| `dataset.py` name | Expected folder | Source |
|---|---|---|
| `mvtec_ad` | `MVTec AD/` | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| `mvtec_ad_2` | `mvtec_ad_2/` | [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) |
| `severstal` | `severstal-steel-defect-detection/` | [Severstal Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/data) |
| `neu_det` | `NEU-DET/` | [NEU Surface Defect](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| `dagm2007` | `DAGM_KaggleUpload/` | [DAGM 2007](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection) |
| `defect_spectrum` | `Hug/` | [DefectSpectrum](https://huggingface.co/datasets/DefectSpectrum/Defect_Spectrum) |

### Expected folder structures

```
root/
├── MVTec AD/                      # mvtec_ad
│   └── <category>/
│       ├── train/good/
│       ├── test/<defect_type>/
│       └── ground_truth/<defect_type>/  # *_mask.png
├── mvtec_ad_2/                    # mvtec_ad_2
│   └── <category>/
│       ├── train/good/
│       ├── validation/good/
│       └── test_public/
│           ├── bad/
│           ├── good/
│           └── ground_truth/bad/       # *_mask.png
├── severstal-steel-defect-detection/ # severstal
│   ├── train.csv                  # ImageId,ClassId,EncodedPixels
│   └── train_images/
├── NEU-DET/                       # neu_det
│   ├── train/
│   │   ├── annotations/           # Pascal VOC XML
│   │   └── images/<class>/
│   └── validation/
│       ├── annotations/
│       └── images/<class>/
├── DAGM_KaggleUpload/             # dagm2007
│   └── Class{1..10}/
│       ├── Train/
│       │   ├── *.PNG
│       │   └── Label/*_label.PNG
│       └── Test/
│           ├── *.PNG
│           └── Label/*_label.PNG
└── Hug/                           # defect_spectrum
    ├── DS-MVTec/<category>/{image,mask}/
    ├── DS-DAGM/{image,mask}/
    ├── DS-Cotton-Fabric/{image,mask}/
    └── DS-VISION/<category>/{image,mask}/
```

## Quick start

```python
from dataset import UniversalIndustrialDataset

ds = UniversalIndustrialDataset(
    root_dir="G:/Dataset",
    dataset_name="mvtec_ad",
    split="train",
)
item = ds[0]
# item["image"]         : (3, 1024, 1024) float32  ImageNet-normalized
# item["mask"]          : (1, 1024, 1024) float32  binary
# item["box_prompt"]    : (4,) int64               [x1,y1,x2,y2]
# item["class_label"]   : int                      per-dataset label
# item["source_dataset"]: str
```

### Map labels to unified 8-class vocabulary

```python
from label_mapping import map_dataset_item

item = map_dataset_item(item)
# item["class_label"]  : int in [0..7]  unified vocabulary
```

## Architecture

```
Image (B, 3, 1024, 1024)
  │
  ▼
SAM ViT-B Image Encoder ── (frozen, 24 injected IndustrialAdapters)
  │
  ▼  (B, 256, 64, 64)
Prompt Encoder  (box prompt from mask)
  │
  ▼
Mask Decoder (TwoWayTransformer)
  ├──▶ mask_logits (B, 1, 256, 256)
  │
  └──▶ IoU token ──▶ DefectClassifierHead ──▶ class_logits (B, 8)
```

## Components

| File | Purpose |
|---|---|
| `dataset.py` | UniversalIndustrialDataset — 6 dataset parsers, box prompts, splits |
| `adapters.py` | IndustrialAdapter: bottleneck (768→64→768) + skip, zero-init up-projection |
| `model_setup.py` | Freeze encoder, inject 24 adapters (2 per block), gradient isolation verification |
| `classifier_head.py` | DefectClassifierHead (256→128→8) + 8-class vocabulary |
| `model.py` | IndustrialSAM — multi-task wrapper with IoU-token hook |
| `label_mapping.py` | Map per-dataset labels → unified 8-class vocabulary |

## Run tests

```bash
python dataset.py           # sanity-check all 6 datasets
python dataset.py --test    # parser unit tests (no real data)
python model_setup.py sam_vit_b_01ec64.pth  # verify adapter injection
python model.py sam_vit_b_01ec64.pth        # shape-check smoke test
python label_mapping.py                     # verify label map integrity
```

## Notes

- All images are **letterboxed** to 1024×1024 (aspect ratio preserved, zero-padded), not stretched.
- DAGM2007 masks are **coarse elliptical labels**, not pixel-precise ground truth.
- NEU-DET masks are **bounding-box derived**, not pixel-precise.
- Severstal and MVTec AD 2 use **custom 70/15/15 stratified splits** (no public test labels).
- 95.66 % of total parameters frozen; 2.4 M trainable adapter params + 33.9 K classifier head.
- The classifier head's 8-class vocabulary is unified across all datasets; use `label_mapping.py` to convert per-dataset labels during training.
