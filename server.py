from flask import Flask, request, jsonify, send_file
import segmentation_models as sm
import gdown
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

# ===============================
# Config TensorFlow / Flask
# ===============================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# ===============================
# Paramètres du modèle
# ===============================
BACKBONE = 'resnet50'
IMG_SIZE = (256, 512)
NUM_CLASSES = 8
MODEL_FILE_ID = "1SWEk8PpN_oM-jIAQ4Sy1-C2pLbCDfvYV"
MODEL_PATH = "bestt_model.h5"

# ===============================
# Couleurs pour les classes
# ===============================
class_colors = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [152, 251, 152],
    [70, 130, 180], [0, 0, 0]
]

# ===============================
# Téléchargement du modèle depuis Google Drive
# ===============================
def download_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        print("📥 Téléchargement du modèle depuis Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print("✅ Téléchargement terminé !")
    else:
        print("✅ Modèle déjà disponible.")

download_model_from_drive(MODEL_FILE_ID, MODEL_PATH)

# ===============================
# Fonctions de perte et métriques
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
    return model

print("⏳ Chargement du modèle...")
model = load_segmentation_model()
print("✅ Modèle chargé avec succès !")

# ===============================
# Fonction pour coloriser le masque
# ===============================
def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        rgb[mask == class_id] = color
    return rgb

# ===============================
# Endpoint pour la segmentation
# ===============================
@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    original_size = img.size

    # Prétraitement
    img_resized = img.resize((IMG_SIZE[1], IMG_SIZE[0]))  # largeur, hauteur
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Segmentation
    pred = model.predict(img_array)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)

    color_mask = decode_segmap(mask)

    # Redimensionner à la taille originale
    mask_img = Image.fromarray(color_mask)
    mask_img = mask_img.resize(original_size, Image.NEAREST)

    # Retourner l'image PNG
    img_bytes = io.BytesIO()
    mask_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')

# ===============================
# Lancement de l'application
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
