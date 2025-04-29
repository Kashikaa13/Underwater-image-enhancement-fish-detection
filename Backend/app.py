from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import io
import os
import numpy as np
import cv2
import base64

# Import Generator model
from models.model import GeneratorUSRGAN

# Import YOLOv8
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SRGAN model
model = GeneratorUSRGAN()
model.load_state_dict(torch.load("checkpoints/srgan_finetuned_ssim2.pth", map_location=device))
model.to(device)
model.eval()

# Load YOLOv8 detection model
yolo_model = YOLO("checkpoints/best.pt")

# Image transformation
transform_tensor = transforms.Compose([
    transforms.ToTensor()
])

# Color enhancement (CLAHE)
def enhance_colors_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Boost saturation
def boost_saturation(pil_image, factor=1.5):
    enhancer = ImageEnhance.Color(pil_image)
    return enhancer.enhance(factor)

@app.route('/')
def index():
    return jsonify({"message": "Backend running"}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    try:
        # Read input
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        # Enhance using SRGAN
        input_tensor = transform_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        sr_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_image = ((sr_image + 1) / 2 * 255).astype("uint8")

        # CLAHE
        clahe_img = enhance_colors_clahe(sr_image)

        # Saturation boost
        pil_img = Image.fromarray(clahe_img)
        saturated_img = boost_saturation(pil_img, factor=1.5)
        saturated_img = boost_saturation(saturated_img, factor=1.5)

        # Convert to OpenCV BGR
        enhanced_cv2 = cv2.cvtColor(np.array(saturated_img), cv2.COLOR_RGB2BGR)

        # Run YOLOv8 detection
        results = yolo_model(enhanced_cv2, verbose=False)[0]

        # Plot bounding boxes
        detection_img = results.plot()

        # Encode both enhanced and detection images to base64
        enhanced_buffer = io.BytesIO()
        saturated_img.save(enhanced_buffer, format='JPEG')
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')

        _, detection_encoded = cv2.imencode('.jpg', detection_img)
        detection_base64 = base64.b64encode(detection_encoded.tobytes()).decode('utf-8')

        return jsonify({
            "enhanced_image": enhanced_base64,
            "detected_image": detection_base64
        })

    except Exception as e:
        app.logger.error(f"Processing failed: {str(e)}")
        return jsonify({'error': 'Image processing failed', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
