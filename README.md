# Multi-Format Digital Weighing Scale OCR Engine (Production-Grade)

## 📌 Project Overview
This repository contains the end-to-end evaluation, training, and deployment pipeline for a robust computer vision system designed to read varied digital weighing scale displays. 

Reading 7-segment industrial LED arrays and reflective LCD screens presents severe machine vision challenges—including discontinuous character strokes, sub-pixel decimal points, plastic bezel glares, and perspective distortions. This project documents the architectural evolution from brittle off-the-shelf OCR engines to a production-grade, dual-mode hybrid architecture that achieves near-100% operational reliability in live mobile application integrations.

## 🏗️ Dual-Mode Pipeline Architecture
To eliminate structural vulnerabilities, the pipeline employs a decoupled architecture optimized inside the production deployment script (`05_Cloud_API_Deployment/main.py`):

1. **Primary Mode (Fully Automated):**
   * **ROI Detection:** A fine-tuned YOLOv8 model localizes the display boundaries.
   * **Spatial Trimming:** A dynamic 3% boundary shave isolates active digits, neutralizing reflection noise from surrounding bezels.
2. **Fallback Mode (YOLO Bypass):**
   * When severe viewing angles or un-indexed scale designs reduce object detection confidence, the system safely switches to a manual close-crop loop, routing user-defined coordinates directly to the sequence engine to prevent failure propagation.
3. **Sequence Recognition:** A Vision Transformer (PARSeq) parses the processed frame using parallel attention decoding.
4. **Heuristic Post-Processing:** Format-aware string processing rules reconstruct sub-pixel decimal drops dynamically based on device-specific form factor constraints.

## 📂 Repository Structure
* `01_Baseline_Benchmarking/` * Evaluation metrics for Tesseract, EasyOCR, SSOCR, and raw PARSeq configurations. Establishes the baseline vulnerability profile.
* `02_PARSeq_Finetuning/` * Quantifies training ceilings when forcing a transformer network to handle localization and recognition simultaneously.
* `03_YOLO_PARSeq_Pipeline/` * Introduction of decoupled spatial object routing, elevating processing benchmarks.
* `04_Final_Optimized_Pipeline/` * Implementation of photometric gradient protection, spatial trimming logic, and heuristic post-processing layers.
* `05_Cloud_API_Deployment/` * Production-ready `main.py` script, Dockerfile configurations, and microservice infrastructure utilizing FastAPI for client-side integration.

## 📊 Performance & Evolution Roadmap
The system tracks **Exact Match Accuracy** (requiring flawless digit and decimal extraction) and **Character Error Rate (CER)**.

| Development Phase | Core Architecture Stack | Exact Match Accuracy | CER | Operational Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Baselines** | PARSeq (Raw Unprocessed RGB) | 26.00% | 58.77% | Highly Unstable |
| **Phase 2: Decoupled ML** | YOLOv8 + PARSeq Transformer | 65.66% | 10.44% | Erratic Decimals |
| **Phase 3: Core Optimized**| YOLOv8 + PARSeq + Spatial Trimming | 90.91% | 1.68% | Production Baseline |
| **Phase 4: Client Release** | YOLO + PARSeq + Dual-Mode Manual Bypass | **~100%** | **<1.00%**| **Live / Deployment Ready** |

### Multi-Domain Verification Proofs
The updated backend successfully generalizes across completely distinct hardware layers under live settings:
* **Industrial Green LED Displays (LP7510):** High-confidence localization (up to 92.9% detector confidence) with full floating-point precision formatting.
* **Benchtop Red LED Arrays:** Flawless character recognition under severe out-of-focus sensor capture.
* **Commercial Reflective LCD Screens:** Invariant extraction of black numeric characters against green-backlit reflective backgrounds under angular skews.

## ☁️ Cloud Deployment Configuration
The inference pipeline is served via a FastAPI backend wrapped inside a Docker container optimized for headless Linux execution dependencies. 
* **Live Deployment Space:** https://princemridul-sol9x-scale-ocr-api.hf.space/predict
* **Inference Payload:** Accepts `multipart/form-data` image uploads and outputs strict JSON strings containing real-time extracted mass, stability states, and system confidence parameters.
