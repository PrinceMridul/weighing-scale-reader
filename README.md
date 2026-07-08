# AI Weighing Scale Reader (LP7510)

## 📌 Project Overview
This repository contains the evaluation, training, and optimization pipeline for an end-to-end Optical Character Recognition (OCR) system designed to read LP7510 industrial digital weighing scales. 

Reading industrial 7-segment LED displays presents unique machine vision challenges, including discontinuous character strokes, sub-pixel decimal points, environmental glare, and background noise. This project documents the architectural evolution from benchmarking off-the-shelf OCR engines to deploying a custom, hybrid machine learning pipeline that achieves **90.91% Exact Match Accuracy**.

## 🏗️ Pipeline Architecture (Final Optimized)
The production-ready architecture utilizes a decoupled localization and recognition framework wrapped in a business logic layer:
1. **Region of Interest (ROI) Detection:** A custom-trained YOLOv8 model extracts the bounding box of the digital display.
2. **Spatial Trimming:** A dynamic 3% boundary shave isolates the lit LEDs, eliminating phantom digits caused by plastic bezel glare.
3. **Sequence Recognition:** A pre-trained Vision Transformer (PARSeq) decodes the image crop into a raw string sequence.
4. **Heuristic Formatting:** Deterministic regex rules reconstruct sub-pixel decimal drops based on the known LP7510 constraint format (`XX.XXX`).

---

## 📂 Repository Structure & Engineering Journey

The repository is structured chronologically to reflect the research and development phases of the pipeline:

* **[`01_Baseline_Benchmarking/`](./01_Baseline_Benchmarking)** * Evaluation of Tesseract, EasyOCR, SSOCR, and PARSeq on raw and OpenCV-preprocessed environmental captures. Proved that aggressive binarization destroys the natural RGB gradients required by Vision Transformers.
* **[`02_PARSeq_Finetuning/`](./02_PARSeq_Finetuning)** * Initial attempts to fine-tune the PARSeq backbone directly on LMDB datasets. The model stalled at ~14.99% accuracy due to the simultaneous burden of spatial localization and sequence recognition.
* **[`03_YOLO_PARSeq_Pipeline/`](./03_YOLO_PARSeq_Pipeline)** * The strategic pivot to a two-stage architecture. Introducing YOLOv8 for ROI detection bumped accuracy to 65.66%, but the model still struggled with bounding box edge artifacts and dropped sub-pixel decimals.
* **[`04_Final_Optimized_Pipeline/`](./04_Final_Optimized_Pipeline)** * The production pipeline. Replaced destructive photometric preprocessing with spatial trimming and domain-specific post-processing, pushing the exact match accuracy to >90%.

---

## 📊 Performance Evolution

The primary metric for success is **Exact Match Accuracy**, requiring the model to capture all digits and the precise decimal placement perfectly. Character Error Rate (CER) was tracked to monitor baseline token health.

| Development Phase | Architecture | Exact Match | CER |
| :--- | :--- | :--- | :--- |
| *Phase 1: Baselines* | PARSeq (Raw RGB) | 26.00% | 58.77% |
| *Phase 2: Object Detection*| YOLOv8 + PARSeq | 65.66% | 10.44% |
| **Phase 3: Production** | **YOLOv8 + PARSeq + Heuristics** | **90.91%** | **1.68%** |

---
## ☁️ Cloud Deployment (Production API)
To support seamless integration with mobile client applications, the verified pipeline has been containerized and deployed as a scalable REST API.

* **Architecture:** FastAPI backend containerized with Docker.
* **Environment:** Optimized with Linux-based graphics dependencies for headless inference on Hugging Face Spaces.
* **Endpoint:** [https://princemridul-sol9x-scale-ocr-api.hf.space/predict](https://princemridul-sol9x-scale-ocr-api.hf.space/predict)
* **Usage:** Accepts `multipart/form-data` image uploads and returns real-time JSON inference results.

See the [`05_Cloud_API_Deployment/`](./05_Cloud_API_Deployment) directory for the `Dockerfile`, `main.py`, and infrastructure configuration files.

---
## 🚀 Getting Started

### 1. Installation
Clone the repository and install the global dependencies.
```bash
git clone [https://github.com/avtechfin/ai-weighing-scale-reader.git](https://github.com/avtechfin/ai-weighing-scale-reader.git)
cd ai-weighing-scale-reader
pip install -r requirements.txt
```
### 2. Dataset Access
The image datasets utilized in these experiments are managed via HuggingFace `datasets` and Roboflow. See the individual module `src` files for automatic dataset download scripts. Ensure you have the necessary API keys configured in your environment.

### 3. Running the Final Pipeline
Navigate to `04_Final_Optimized_Pipeline/src/` to execute the verified evaluation notebook against the Gold Standard verification dataset.

```python
# Core Inference Snippet Example
img = cv2.imread('scale_display.jpg')

# 1. Detect
results = yolo_model(img)
boxes = results[0].boxes.xyxy.cpu().numpy()

# 2. Crop & Clean
x1, y1, x2, y2 = map(int, boxes[0])
clean_crop = spatial_trimming(img[y1:y2, x1:x2])

# 3. Read & Format
raw_text = parseq_inference(clean_crop)
final_reading = post_process_prediction(raw_text)
```
## 🔮 Future Development
The current hybrid pipeline serves as a highly viable baseline for automated data entry. To achieve >99% generalization across *any* scale format (removing the hardcoded `XX.XXX` heuristics), future iterations will focus on generating a synthetic LMDB dataset consisting strictly of cropped green LED arrays. This will be used to explicitly fine-tune the PARSeq attention heads to natively recognize sub-pixel decimals.
