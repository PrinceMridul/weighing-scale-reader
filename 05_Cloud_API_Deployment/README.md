# FastAPI Cloud Deployment

This directory contains the production-grade deployment configuration for the LP7510 weighing scale reader. The pipeline is containerized and hosted as a REST API on Hugging Face Spaces.

## Architecture
- **Framework:** FastAPI (High-performance asynchronous API)
- **Containerization:** Docker (Optimized with custom Linux graphics drivers for OpenCV)
- **Model Loading:** TorchHub with dynamic trust for PARSeq
- **Deployment:** Hugging Face Spaces (CPU-optimized)

## Deployment Stack
- `Dockerfile`: Multi-stage blueprint for Linux environment setup.
- `requirements.txt`: Curated dependency list using `opencv-python-headless` to eliminate GUI dependencies.
- `main.py`: Inference engine utilizing YOLOv8 for ROI detection and PARSeq for text recognition, with integrated regex-based error correction for decimal scaling.

## API Usage
The API endpoint is hosted at: `https://princemridul-sol9x-scale-ocr-api.hf.space/predict`

**POST Request:**
- `file`: Image file (multipart/form-data)
- Returns: JSON response with `reading`, `yolo_box_found`, and `raw_parseq` output.
