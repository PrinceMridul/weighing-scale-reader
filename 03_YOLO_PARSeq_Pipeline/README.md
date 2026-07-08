# Module 03: Two-Stage Pipeline (YOLOv8 ROI Detection + PARSeq)

## Overview
Following the baseline benchmarking in Module 01, it was determined that attempting to read full, uncropped environmental captures of industrial scales introduces too much background noise. 

This module implements a two-stage Optical Character Recognition (OCR) pipeline:
1. **Region of Interest (ROI) Detection:** A custom-trained YOLOv8 model (`best_yolo.pt`) detects the 7-segment LED display and outputs bounding box coordinates.
2. **Text Recognition:** The image is spatially cropped using the YOLO coordinates and passed to the pre-trained Vision Transformer (PARSeq) to extract the text.

## Pipeline Architecture
The pipeline is designed to decouple localization from recognition. By utilizing YOLOv8 for spatial isolation, the PARSeq Transformer is fed clean, tightly bounded crops, drastically reducing the environmental noise it needs to process.

### Implementation Workflow
1. Load YOLOv8 and official pre-trained PARSeq weights (`baudm/parseq`).
2. Run YOLO inference to extract the `xyxy` bounding box with the highest confidence score.
3. Spatially crop the raw BGR image array.
4. Transform the crop (Bicubic Resize to 32x128, Normalize) and pass it to PARSeq for greedy decoding.

```python
def run_pipeline(image_path, yolo_model, parseq_model, img_transform, device):
    image_bgr = cv2.imread(image_path)
    
    # 1. Detect ROI
    xyxy, yolo_conf = detect_roi(image_bgr, yolo_model)
    
    # 2. Spatial Crop
    crop = image_bgr[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    
    # 3. Recognition
    text, parseq_conf = recognize_text(crop, parseq_model, img_transform, device)
    
    return {"predicted_text": text, "yolo_conf": yolo_conf, "parseq_conf": parseq_conf}
```
## Benchmark Results
Evaluated on the full human-verified Gold Standard dataset.

| Model Architecture | Exact Match Accuracy | Character Error Rate (CER) |
| :--- | :--- | :--- |
| **YOLOv8 + PARSeq (Base)** | **65.66%** | **10.44%** |

*(Note: The provided notebook demonstrates a verification run on a 10-image subset yielding an 80.00% exact match, while the global evaluation across the entire validation dataset stabilizes at 65.66%).*

## Analysis & Identified Failure Modes
While isolating the ROI with YOLOv8 significantly improved accuracy compared to the raw baselines in Module 01, the pipeline stalled at ~65% due to two specific integration errors between the models:

1. **The Bezel Artifact (Phantom Digits):** YOLOv8 bounding boxes occasionally include a few pixels of the dark plastic bezel surrounding the LEDs. Because PARSeq is highly sensitive to edge gradients, it frequently hallucinates this dark edge as a leading 7, 1, or 2 (e.g., Truth: 82.700 → Prediction: 782700).

2. **Dropped Decimals:** Sub-pixel LED decimal points are still occasionally dropped due to glare within the YOLO crop boundary.

**Next Steps:** These specific formatting and bounding box noise issues are resolved in `04_Final_Optimized_Pipeline` using spatial edge shaving and deterministic business logic.
