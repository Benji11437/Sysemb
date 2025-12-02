from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

IMG_HEIGHT = 256
IMG_WIDTH = 512

model = load_model("bests_model.h5", compile=False)

class_colors = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [152, 251, 152],
    [70, 130, 180], [0, 0, 0]
]

def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        rgb[mask == class_id] = color
    return rgb

@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    original_size = img.size

    # Prétraitement
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Segmentation
    pred = model.predict(img_array)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    color_mask = decode_segmap(mask)

    # Retour à la taille d'origine
    mask_img = Image.fromarray(color_mask)
    mask_img = mask_img.resize(original_size, Image.NEAREST)

    # Conversion en binaire
    img_bytes = io.BytesIO()
    mask_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
