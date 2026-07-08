# Module 02: PARSeq Fine-Tuning (Initial Attempt)

## Overview
Following the baseline benchmarking in Module 01, this module details the initial attempt to fine-tune the PARSeq (Permuted Autoregressive Sequence) Vision Transformer directly on our custom dataset of LP7510 weighing scale images. 

The working hypothesis was that fine-tuning the model natively would allow the ViT backbone to learn the sub-pixel features of 7-segment LED displays and ignore environmental glare without needing complex pre-processing.

## Training Configuration & Dataset Preparation
To utilize the official PARSeq training framework (`strhub`), the raw dataset was split and converted into the high-performance LMDB format required by the model.

* **Architecture:** `parseq-tiny`
* **Training Set:** 2,666 images
* **Validation Set:** 667 images
* **Framework:** PyTorch Lightning
* **Epochs:** 100
* **Precision:** 16-bit Mixed Precision (AMP)

### LMDB Conversion Script
```python
# Converting raw images to LMDB format for strhub compatibility
!python tools/create_lmdb_dataset.py data/train/real data/train/train_labels.txt data/train/real
!python tools/create_lmdb_dataset.py data/val data/val/val_labels.txt data/val
```
## Results
The fine-tuning experiment was executed for the full 100 epochs, but the model failed to converge to a production-ready state, plateauing very early in the training cycle[cite: 3].

| Metric | Result |
| :--- | :--- |
| **Max Validation Accuracy (Exact Match)** | **~14.99%** |
| **Normalized Edit Distance (NED)** | 9.78 |

*Best Checkpoint:* `epoch=2-step=21-val_accuracy=0.1499-val_NED=9.7826.ckpt`[cite: 3]

## Analysis: Why did it underperform?
The fine-tuning stalled at ~15% accuracy due to a fundamental architectural bottleneck: **Localization vs. Recognition**. 

Because the raw dataset images contained significant amounts of environmental background noise (factory settings, glare, the plastic scale casing), the Transformer struggled to simultaneously learn *where* the text was and *what* the text was. The attention heads exhausted computational power attempting to decode background artifacts instead of focusing purely on the 7-segment LEDs.

## Strategic Pivot
Instead of brute-forcing the Transformer to handle both spatial localization and character recognition, the decision was made to halt this fine-tuning approach and pivot to a decoupled, two-stage architecture:
1. **Stage 1 (Module 03):** Introduce a dedicated Object Detection model (YOLOv8) to handle strict Region of Interest (ROI) extraction.
2. **Stage 2:** Pass only those tightly bounded, clean crops into the baseline PARSeq model.

**Future Development:** Now that the hybrid YOLOv8 + PARSeq pipeline has successfully stabilized at >90% accuracy utilizing business logic heuristics (see Module 04), this fine-tuning module will be revisited. Future iterations will involve generating a new LMDB dataset utilizing *only the clean YOLO crops* to fine-tune the PARSeq backbone, bridging the final gap to 99%+ accuracy without hardcoded formatting rules.
