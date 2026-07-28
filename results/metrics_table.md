# Evaluation Results

## Overall Metrics

| Metric | Value |
|--------|-------|
| mIoU | 0.2857 |
| Dice | 0.3888 |
| Accuracy | 0.5311 |
| Macro F1 | 0.0883 |

## Per-Dataset Segmentation

| Dataset | mIoU | Dice | Samples |
|---------|------|------|---------|
| dagm2007 | 0.4267 | 0.5826 | 1054 |
| defect_spectrum | 0.3101 | 0.3884 | 161 |
| mvtec_ad | 0.6183 | 0.7334 | 191 |
| mvtec_ad_2 | 0.2943 | 0.3745 | 102 |
| severstal | 0.0819 | 0.1365 | 1064 |

## Per-Class Classification

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Flawless | 0.0000 | 0.0000 | 0.0000 | 0 |
| Surface Scratch | 0.0000 | 0.0000 | 0.0000 | 794 |
| Structural Crack | 0.2500 | 0.0065 | 0.0127 | 153 |
| Hole / Puncture | 0.0000 | 0.0000 | 0.0000 | 145 |
| Inclusion | 0.0000 | 0.0000 | 0.0000 | 52 |
| Missing Component | 0.0000 | 0.0000 | 0.0000 | 0 |
| Discoloration / Stain | 0.0000 | 0.0000 | 0.0000 | 60 |
| Geometric Deformation | 0.5315 | 0.9978 | 0.6936 | 1368 |