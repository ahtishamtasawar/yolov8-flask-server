from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("best.pt")

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
