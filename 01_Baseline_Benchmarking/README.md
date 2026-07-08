# Module 01: Baseline Benchmarking & Preprocessing Experiments

## Overview
The primary objective of this study is to evaluate and compare the performance of distinct Optical Character Recognition (OCR) frameworks on a specialized, cropped dataset of seven-segment Light Emitting Diode (LED) displays. Industrial digit displays present unique recognition challenges due to disconnected character strokes, low contrast, and environmental pixel noise. 

This module tests four distinct architectural paradigms on a 50-image verification subset to establish a baseline before building the final pipeline:
1. **Tesseract (Legacy/Geometric):** Relies on classical line finding and character cell layout analysis.
2. **EasyOCR (Deep Learning Pipeline):** Utilizes a CRAFT text detector and a CRNN (ResNet + LSTM) sequence recognizer.
3. **PARSeq (Vision Transformer):** A state-of-the-art model leveraging a pure Vision Transformer (ViT) architecture.
4. **SSOCR:** A domain-specific C-tool designed specifically for seven-segment displays.

## Evaluation Metrics
Performance is validated using four critical metrics:
* **Exact Match Accuracy:** The percentage of images where the predicted string perfectly matches the ground truth label.
* **Character Accuracy:** The normalized accuracy of individual characters predicted correctly.
* **Character Error Rate (CER):** Calculated via the Levenshtein distance, measuring the minimum edit operations required to transform the predicted string into the truth string (lower is better).
* **Avg Inference Time:** The mean computational latency required to process a single image frame, measured in milliseconds (ms).

---

## Experiment 1: Baseline Models & Domain-Specific Tools
Models were fed a mix of Raw RGB inputs and baseline domain-specific configurations to test out-of-the-box generalization. Tesseract was tested with its required OpenCV preprocessing (as it cannot natively handle raw dark-mode LEDs well), while EasyOCR and PARSeq processed the raw environmental crops.

### Results
| Model Architecture | Exact Match Accuracy | Character Accuracy | CER | Avg Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Tesseract (OpenCV Preprocessed)** | 0.08 | 0.2567 | 0.7483 | 134.26 |
| **SSOCR (Domain-Specific C-Tool)** | 0.00 | 0.0000 | 1.0000 | 3.76 |
| **EasyOCR (Raw RGB)** | 0.12 | 0.4720 | 1.1300 | 39.48 |
| **PARSeq (Raw RGB)** | **0.26** | **0.4920** | **0.5877** | 21.04 |

### Analysis
* **PARSeq Dominance:** PARSeq significantly outperformed the other models across all evaluated accuracy metrics while maintaining a highly efficient latency profile (21.04 ms).
* **SSOCR Failure:** Despite being explicitly designed for seven-segment displays, the legacy SSOCR C-tool completely failed (0.00 Exact Match) on environmental captures due to its inability to handle background noise and glare, though it was extremely fast (3.76 ms).
* **EasyOCR Hallucinations:** Despite decent character tracking (0.4720), EasyOCR's CER spiked heavily to 1.1300. The Connectionist Temporal Classification (CTC) decoding mechanism suffered from sequence length misalignment, introducing repetitive token insertions.

---

## Experiment 2: Global OpenCV Preprocessing (Ablation Study)
To address the disconnected strokes inherent to 7-segment LEDs, a global OpenCV preprocessing pipeline was uniformly applied to the core OCR engines. The images were converted to grayscale, subjected to Otsu's Thresholding (forced pure black/white), and treated with a Morphological Closing kernel (3x3) to physically bridge the LED gaps before inference.

### Results
| Model Architecture | Exact Match Accuracy | Character Accuracy | CER | Avg Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Tesseract (OpenCV)** | 0.08 | 0.2567 | 0.7483 | 162.93 |
| **EasyOCR (OpenCV)** | 0.08 | 0.5070 | 0.5470 | 23.05 |
| **PARSeq (OpenCV)** | **0.26** | **0.6027** | **0.5393** | 28.08 |

### Analysis & Conclusion
* **Transformer Resistance:** Morphological closing artificially bridged the 7-segment gaps, which slightly improved PARSeq's Character Accuracy (from 0.4920 to 0.6027). However, the Exact Match Accuracy remained entirely stagnant at 0.26. 
* **Feature Destruction:** Pre-trained deep learning models, particularly Vision Transformers like PARSeq, extract features using natural color gradients and sub-pixel anti-aliasing. Forcing the image into a binary threshold destroys these natural gradients, effectively capping the model's ability to read the exact decimal placements.

**Decision:** The pipeline will proceed using raw RGB image inputs and the PARSeq architecture, abandoning destructive global binarization techniques in favor of spatial cropping strategies later in the pipeline.

---

## Core Implementation Snippets

**OpenCV Morphological Closing Function**
```python
import cv2
import numpy as np
from PIL import Image

def apply_opencv_preprocessing(pil_img):
    img_cv = np.array(pil_img.convert('L'))
    
    # Otsu's Thresholding
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    final_img = cv2.bitwise_not(closed)
    return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB))
