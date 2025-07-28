
from flask import Flask, request, send_file, render_template
from realesrgan import RealESRGAN
from PIL import Image
import os
import uuid
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image uploaded.', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file.', 400

    input_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + file.filename)
    file.save(input_path)

    # Load image
    img = Image.open(input_path).convert('RGB')

    # Load model
    model_path = 'models/RealESRGAN_x4plus.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(model_path)

    sr_image = model.predict(img)

    output_path = os.path.join(OUTPUT_FOLDER, 'mejorado_' + os.path.basename(input_path))
    sr_image.save(output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
