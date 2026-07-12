# Multi-Format Digital Weighing Scale OCR Engine (Production-Grade)

## 📌 Project Overview
This repository contains the end-to-end evaluation, training, and deployment pipeline for a robust computer vision system designed to read varied digital weighing scale displays. 

Reading 7-segment industrial LED arrays and reflective LCD screens presents severe machine vision challenges—including discontinuous character strokes, sub-pixel decimal points, plastic bezel glares, and perspective distortions. This project documents the architectural evolution from brittle off-the-shelf OCR engines to a production-grade, dual-mode hybrid architecture that achieves near-100% operational reliability in live mobile and cloud application integrations.

## 🏗️ Dual-Mode Pipeline Architecture
To eliminate structural vulnerabilities, the pipeline employs a decoupled architecture optimized for both cloud deployment (`05_Cloud_API_Deployment`) and on-device edge execution (`07_YOLO_Augmented_Finetuning`):

1. **Primary Mode (Fully Automated):**
   * **ROI Detection:** A fine-tuned, generalized YOLOv8 model localizes display boundaries, seamlessly ignoring GPS watermarks and background debris.
   * **Adaptive Preprocessing:** Dynamic channel extraction and morphological closing filter out sunlight glare and heal dirt-covered LED segments.
   * **Sequence Recognition:** A Vision Transformer (PARSeq) parses the processed frame using parallel attention decoding.
2. **Fallback Mode (YOLO Bypass):**
   * When severe viewing angles or un-indexed scale designs reduce object detection confidence, the system safely switches to a manual close-crop loop, routing user-defined coordinates directly to the sequence engine to prevent failure propagation.
3. **Heuristic Post-Processing:** Format-aware string processing rules reconstruct sub-pixel decimal drops dynamically and strip out artifact hallucinations (e.g., artificial leading zeros).

## 📂 Repository Structure & Engineering Journey
The repository is structured chronologically to reflect the research and development phases of the pipeline:

* **[`01_Baseline_Benchmarking/`](./01_Baseline_Benchmarking)** * Evaluation of Tesseract, EasyOCR, SSOCR, and PARSeq on raw and OpenCV-preprocessed environmental captures. Proved that aggressive binarization destroys the natural RGB gradients required by Vision Transformers.
* **[`02_PARSeq_Finetuning/`](./02_PARSeq_Finetuning)** * Initial attempts to fine-tune the PARSeq backbone directly on LMDB datasets. The model stalled at ~14.99% accuracy due to the simultaneous burden of spatial localization and sequence recognition.
* **[`03_YOLO_PARSeq_Pipeline/`](./03_YOLO_PARSeq_Pipeline)** * The strategic pivot to a two-stage architecture. Introducing YOLOv8 for ROI detection bumped accuracy to 65.66%, but the model still struggled with bounding box edge artifacts and dropped sub-pixel decimals.
* **[`04_Final_Optimized_Pipeline/`](./04_Final_Optimized_Pipeline)** * The baseline production pipeline. Replaced destructive photometric preprocessing with spatial trimming and domain-specific post-processing, pushing the exact match accuracy on green-scale industrial displays to >90%.
* **[`05_Cloud_API_Deployment/`](./05_Cloud_API_Deployment)** * Production-ready FastAPI API backend, Dockerfile configurations, and microservice infrastructure optimized for headless cloud execution and dynamic client testing.
* **[`06_Heuristic_Red_LED_Bypass/`](./06_Heuristic_Red_LED_Bypass)** * A zero-training fallback experiment attempting to handle red LED domain shift via pure computer vision (HSV color masking + hardcoded watermarking "Guillotine" cuts). Proved to be 93.7% accurate but highly brittle to camera angular skews.
* **[`07_YOLO_Augmented_Finetuning/`](./07_YOLO_Augmented_Finetuning)** * Deep learning domain adaptation. Artificially multiplied a scarce 16-image red scale dataset to 64 images via geometric/photometric augmentations. Achieved a flawless `0.995` mAP50 on YOLO fine-tuning and successfully exported to a zero-latency TorchScript (`.ptl`) edge model for mobile.

## 📊 Performance & Evolution Roadmap
The system tracks **Exact Match Accuracy** (requiring flawless digit and decimal extraction) and **Character Error Rate (CER)**.

| Development Phase | Core Architecture Stack | Exact Match Accuracy | CER | Operational Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Baselines** | PARSeq (Raw Unprocessed RGB) | 26.00% | 58.77% | Highly Unstable |
| **Phase 2: Decoupled ML** | YOLOv8 + PARSeq Transformer | 65.66% | 10.44% | Erratic Decimals |
| **Phase 3: Core Optimized**| YOLOv8 + PARSeq + Spatial Trimming | 90.91% | 1.68% | Production Baseline |
| **Phase 4: Cloud API** | YOLO + PARSeq + Dual-Mode Manual Bypass | **~100%** | **<1.00%**| **Live / Cloud Ready** |
| **Phase 5: Edge Mobile** | YOLO (TorchScript) + Adaptive OpenCV | **~100%** | **<1.00%**| **Zero-Latency Offline / Live** |

### Multi-Domain Verification Proofs
The updated backend successfully generalizes across completely distinct hardware layers under live settings:
* **Industrial Green LED Displays (LP7510):** High-confidence localization (up to 92.9% detector confidence) with full floating-point precision formatting.
* **Benchtop Red LED Arrays:** Flawless character recognition under severe out-of-focus sensor capture and intrusive GPS camera watermarks.
* **Commercial Reflective LCD Screens:** Invariant extraction of black numeric characters against green-backlit reflective backgrounds under angular skews.

## ☁️ Deployment Configurations

### Cloud API Integration
The inference pipeline is served via a FastAPI backend wrapped inside a Docker container optimized for headless Linux execution dependencies. 
* **Live Deployment Space:** https://princemridul-sol9x-scale-ocr-api.hf.space/predict
* **Inference Payload:** Accepts `multipart/form-data` image uploads and outputs strict JSON strings containing real-time extracted mass, stability states, and system confidence parameters.

### Mobile Edge Integration (Flutter)
For environments with zero internet connectivity (e.g., remote construction gravel yards), the pipeline utilizes a compiled PyTorch Lite (`.ptl`) graph. YOLO localization and adaptive preprocessing run natively on the mobile CPU via `dart:ui` letterboxing, ensuring instant, offline data capture.
