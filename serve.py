from flask import Flask, request, jsonify, send_file
import segmentation_models as sm
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import requests

# ===============================
# Config
# ===============================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

BACKBONE = 'resnet50'
IMG_SIZE = (256, 512)
NUM_CLASSES = 8

# URL publique de ton modèle stocké dans Azure Blob Storage
MODEL_URL = "https://stokagesysemb.blob.core.windows.net/repmodele/model.h5"
MODEL_PATH = "model.h5"

# Classes (CityScapes)
class_colors = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [152, 251, 152],
    [70, 130, 180], [0, 0, 0]
]

# ===============================
# Téléchargement du modèle depuis Azure
# ===============================
def download_model_from_azure(url, output_path):
    """Télécharge le modèle depuis Azure Blob Storage si non présent."""
    if not os.path.exists(output_path):
        print("📥 Téléchargement du modèle depuis Azure Blob Storage...")
        resp = requests.get(url)

        if resp.status_code != 200:
            raise ValueError(f"❌ Impossible de télécharger le modèle : {resp.status_code}")

        with open(output_path, "wb") as f:
            f.write(resp.content)

        print("✅ Modèle téléchargé !")
    else:
        print("✔ Modèle déjà présent localement.")


download_model_from_azure(MODEL_URL, MODEL_PATH)

# ===============================
# Fonctions pertes et métriques
# ===============================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

ce_loss = tf.keras.losses.CategoricalCrossentropy()

def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + ce_loss(y_true, y_pred)

# ===============================
# Chargement du modèle
# ===============================
def load_segmentation_model():
    print("⏳ Chargement du modèle TensorFlow...")
    model = sm.Unet(
        BACKBONE,
        classes=NUM_CLASSES,
        activation='softmax',
        encoder_weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=total_loss,
        metrics=[tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES), dice_coef]
    )

    model.load_weights(MODEL_PATH)
    print("✅ Modèle chargé avec succès !")
    return model


model = load_segmentation_model()

# ===============================
# Utility: Colorisation du masque
# ===============================
def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        rgb[mask == class_id] = color
    return rgb

# ===============================
# Endpoint principal
# ===============================
@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    original_size = img.size

    # Prétraitement
    img_resized = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    pred = model.predict(img_array)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    # Colorisation
    color_mask = decode_segmap(mask)

    # Redimension à la taille d'origine
    mask_img = Image.fromarray(color_mask).resize(original_size, Image.NEAREST)

    # Retour image PNG
    img_bytes = io.BytesIO()
    mask_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype="image/png")



if __name__ == "__main__":
    # Pour Azure, utiliser host=0.0.0.0 et port depuis variable d’environnement
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
