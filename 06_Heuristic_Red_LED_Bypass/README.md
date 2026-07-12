# Phase 6: Heuristic Red LED Bypass (Zero-Training OCR)

## 📌 Objective
To test a rapid, zero-training pipeline capable of reading red 7-segment LED scales without relying on deep learning object detection (YOLO). This phase acts as a benchmark to see if pure computer vision mathematics can replace neural networks for localized hardware reading.

## 🛠️ The Architecture
Instead of using YOLO to find the scale, this pipeline relies on strict environmental assumptions and OpenCV matrix math:

1. **The "Guillotine" Crop:** Assumes the user centers the camera, automatically amputating the bottom 20% of the image array to remove the GPS camera watermark and stray gravel.
2. **HSV Color Masking:** Converts the RGB image to Hue, Saturation, and Value (HSV) to isolate the exact color spectrum of the red LED digits, ignoring daylight glare.
3. **Contour Extraction:** Uses `cv2.findContours` and area filtering to draw a mathematical bounding box around the largest continuous red glowing object in the frame.
4. **PARSeq Text Extraction:** Feeds the mathematically isolated crop into the Vision Transformer.

## 📊 Results & Vulnerabilities
* **Success Rate:** 15 out of 16 raw client images perfectly extracted and read (93.7%).
* **The Flaw:** Proved highly brittle in production. If a user tilted the camera (e.g., Image 5), the scale fell into the bottom 20% "guillotine" zone and was amputated, causing the math to track random red dust instead.

**Conclusion:** Pure heuristic math is fast but too fragile for uncontrolled real-world client captures. This necessitates a shift back to dynamic Machine Learning (YOLO) for spatial localization.
