# Phase 7: YOLO Fine-Tuning & Mobile Edge Export

## 📌 Objective
Addressing the brittleness of the Phase 6 heuristic bypass by returning to a deep learning localization architecture. This phase focuses on rapidly curating a custom dataset to teach YOLOv8 how to dynamically find red scale housings, ignoring background debris, aspect ratio distortions, and GPS watermarks.

## 🛠️ Pipeline Implementation

### 1. Dataset Augmentation (Roboflow)
A severely limited dataset of 16 real-world client images was artificially multiplied to prevent model overfitting.
* **Base Images:** 16 strictly annotated frames (bounding boxes isolating only the plastic display module).
* **Augmentation Multipliers:** Applied rotations (-15° to +15°), brightness variance (±25%), and gaussian blur (up to 1.25px).
* **Final Training Set:** 64 highly diverse domain images.

### 2. Transfer Learning
Utilized the foundational `yolov8n.pt` weights and fine-tuned specifically on the augmented red-scale dataset for 50 epochs. 
* **Metrics:** Achieved `0.995` mAP50 and zero false positives (Precision 99.9%). 
* **Robustness:** The trained network successfully tracks the display at steep angles and completely ignores the hardcoded GPS camera watermarks that broke previous iterations.

### 3. Mobile Edge Export (PyTorch Lite)
The final fine-tuned weights were optimized, stripped of training gradients, and exported to a compiled `.torchscript` format. This allows the localized YOLO inference to run with zero-latency natively on edge mobile CPUs via Flutter's PyTorch Lite runtime.
