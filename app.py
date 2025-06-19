from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

# 🔽 Model setup: Download from Google Drive if not already present
MODEL_PATH = "best.pt"
GOOGLE_DRIVE_FILE_ID = "1zyI5TIqDg3biC5ph6Fq7T5UF8feT3OQc"  # <- Replace with your actual file ID

if not os.path.exists(MODEL_PATH):
    print("🔁 Downloading YOLOv8 model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# 🔄 Load model
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    results = model(image)

    output = []
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            output.append({
                "class": cls,
                "confidence": conf,
                "bbox": xyxy
            })

    return jsonify(output)
