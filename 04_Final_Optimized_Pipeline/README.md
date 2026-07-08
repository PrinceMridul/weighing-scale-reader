# Module 04: Final Optimized Pipeline (YOLOv8 + PARSeq + Heuristics)

## Overview
This module contains the final, optimized Optical Character Recognition (OCR) pipeline for reading LP7510 digital weighing scale displays. The architecture utilizes YOLOv8 for Region of Interest (ROI) extraction and a pre-trained Vision Transformer (PARSeq) for text recognition. 

To bridge the gap between raw ML predictions (~75% accuracy) and production readiness, an extensive ablation study on photometric preprocessing was conducted. Deep learning Vision Transformers require natural RGB gradients for feature extraction; therefore, destructive preprocessing techniques (binarization, aggressive thresholding) were ultimately discarded in favor of spatial trimming and domain-specific string heuristics.

## The Challenge
The initial pipeline evaluation revealed two primary failure modes:
1. **Phantom Leading Digits:** YOLO bounding boxes occasionally captured the dark plastic bezel of the scale. PARSeq hallucinated these edge artifacts as leading digits (e.g., predicting `782.700` instead of `82.700`).
2. **Dropped Decimals:** Sub-pixel LED decimal points were frequently lost due to environmental glare, causing formatting errors (e.g., predicting `87000` instead of `87.000`).

## Iteration & Experimentation History

To resolve these issues, multiple OpenCV preprocessing techniques were applied to the YOLO crops before passing them to PARSeq. The results demonstrated that modifying the natural gradients of the image degraded the Transformer's performance.

### Experiment 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
Attempted to boost the brightness of the LEDs against the dark background by converting to YUV color space and applying CLAHE.
* **Exact Match Accuracy:** 20.20%
* **Character Error Rate (CER):** 17.34%
* **Conclusion:** CLAHE amplified environmental glare and unlit LED segments, adding noise that confused the OCR.

### Experiment 2: Otsu's Thresholding + Morphological Dilation
Attempted to binarize the image to pure black/white and use morphological closing to bridge the gaps in the 7-segment display.
* **Exact Match Accuracy:** 13.13%
* **Character Error Rate (CER):** 19.70%
* **Conclusion:** Binarization destroyed the sub-pixel anti-aliasing features expected by the pre-trained ViT backbone.

### Experiment 3: Gamma Correction + High-Pass Sharpening
Attempted a non-linear brightness adjustment (Gamma = 0.5) paired with a 3x3 sharpening kernel to pronounce the decimal point without binarization.
* **Exact Match Accuracy:** 39.39%
* **Character Error Rate (CER):** 11.45%
* **Conclusion:** While better than thresholding, artificial edge sharpening still distorted the natural gradient distribution.

## Final Implementation: Spatial Trimming & Heuristic Business Logic

Recognizing that PARSeq requires unaltered RGB gradients, the photometric preprocessing was replaced with a hybrid approach combining spatial manipulation and deterministic logic.

**1. Spatial Trimming (`clean_crop`)**
A dynamic 3% edge shave is applied to the YOLO crops. This isolates the lit LEDs and eliminates the plastic bezel shadows responsible for phantom digit hallucinations.

**2. Domain-Specific Post-Processing (`post_process_prediction`)**
Deterministic rules based on the known LP7510 format (XX.XXX) are applied to the raw string output to correct dropped decimals and filter non-numeric noise.

## Final Benchmark Results

Evaluated on the 100-image human-verified Gold Standard dataset utilizing the hybrid pipeline detailed above.

### Execution Output
```text
Using cache found in /root/.cache/torch/hub/baudm_parseq_main
Total Images Evaluated : 99
Exact Matches          : 90
Model Accuracy         : 90.91%
Character Error Rate   : 1.68%
