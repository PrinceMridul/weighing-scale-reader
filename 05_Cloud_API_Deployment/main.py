import io
import cv2
import torch
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms as T

# Initialize the API
app = FastAPI(title="Scale OCR API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {DEVICE}...")

# Load Models (Make sure 'best_yolo.pt' is in the same folder as this script)
yolo = YOLO('best_yolo.pt')
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval().to(DEVICE)

parseq_transform = T.Compose([
    T.Resize((32, 128), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

# ==========================================
# Helper Functions (Your exact logic)
# ==========================================
def clean_crop(img):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Otsu's Thresholding dynamically splits the glowing LEDs from the dark screen background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Find all bright LED pixels
    if cv2.countNonZero(thresh) > 0:
        pts = cv2.findNonZero(thresh)
        # 4. Get the tightest possible box around only the lit text
        x, y, w_box, h_box = cv2.boundingRect(pts)
        
        # Add a tiny 5% padding so PARSeq has a clean border to read
        h_img, w_img = img.shape[:2]
        pad = int(h_img * 0.05) 
        
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w_box + pad)
        y2 = min(h_img, y + h_box + pad)
        
        cropped = img[y1:y2, x1:x2]
    else:
        cropped = img # Fallback if screen is totally dark
        
    # Boost contrast to help PARSeq
    alpha = 1.2 
    beta = 10   
    enhanced = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
    
    # Debugger save
    cv2.imwrite("current_parseq_input.jpg", enhanced)
    
    return enhanced

def filter_by_confidence(raw_str, conf_tensor, threshold=0.40):
    return "".join([char for char, conf in zip(raw_str, conf_tensor) if conf.item() >= threshold])

def post_process_prediction(pred):
    clean = re.sub(r'[^0-9\.]', '', pred)
    if '.' not in clean and len(clean) >= 4:
        clean = clean[:-3] + '.' + clean[-3:]
    return clean

# ==========================================
# The API Endpoint
# ==========================================
@app.post("/predict")
async def predict_scale(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded image from Flutter into OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "Invalid image file format."}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Stage 1: YOLO Detection
        yolo_res = yolo(img_rgb, conf=0.1, verbose=False)
        boxes = yolo_res[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            h, w = img_rgb.shape[:2]
            shaved_crop = img_rgb[0:h, 0:w] # Fallback: Full frame
        else:
            x1, y1, x2, y2 = map(int, boxes[0])
            shaved_crop = clean_crop(img_rgb[y1:y2, x1:x2])

        # 3. Stage 2: OpenCV Dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        shaved_crop = cv2.dilate(shaved_crop, kernel, iterations=1)

        # 4. Stage 3: PARSeq Recognition
        with torch.no_grad():
            tensor = parseq_transform(Image.fromarray(shaved_crop)).unsqueeze(0).to(DEVICE)
            logits = parseq(tensor)
            decoded_strs, confidences = parseq.tokenizer.decode(logits.softmax(-1))

            raw_pred = decoded_strs[0].strip()
            char_confs = confidences[0]

        # 5. Stage 4: Post-Processing
        filtered_pred = filter_by_confidence(raw_pred, char_confs, threshold=0.40)
        final_pred = post_process_prediction(filtered_pred)

        # Return the clean data to the Flutter app!
        return {
            "status": "success",
            "reading": final_pred,
            "debug": {
                "yolo_box_found": len(boxes) > 0,
                "raw_parseq": raw_pred,
                "filtered_parseq": filtered_pred
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
