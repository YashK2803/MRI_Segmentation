import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from PIL import Image
import numpy as np
from raunet2 import RAUNet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for frontend communication

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model initialization
model = RAUNet()
trained_model_path = 'trained_model.pkl'

# Load the trained model weights
try:
    state_dict = torch.load(trained_model_path, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
        print("Trained model weights loaded successfully!")
    else:
        model.load_state_dict(state_dict, strict=False)
        print("Trained model weights loaded with strict=False.")
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Error loading trained model weights: {e}")

def preprocess_image(image):
    target_size = (256, 256)  # Adjust size as needed
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    return image_tensor.unsqueeze(0).to(device)

def postprocess_output(output):
    output = output.squeeze().detach().cpu().numpy()
    output = (output > 0.5).astype(np.uint8) * 255
    return Image.fromarray(output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)

        processed_image = postprocess_output(output)

        # Save the processed image
        output_path = os.path.join(app.root_path, "static", "processed_image.png")
        processed_image.save(output_path)

        return jsonify({'processed_image_url': '/static/processed_image.png'})

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)})

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)