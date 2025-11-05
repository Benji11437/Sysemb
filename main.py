import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import segmentation_models as sm
import gdown
import os

# ===============================
# ⚙️ Configuration Streamlit
# ===============================
st.set_page_config(page_title="Segmentation d'image", layout="wide")
st.title("Application de Segmentation d'Image")

# ===============================
# 📦 Paramètres du modèle
# ===============================
BACKBONE = 'resnet50'
IMG_SIZE = (256, 512)
NUM_CLASSES = 8
MODEL_FILE_ID = "1vjg08BuQTt1nc_eMLLBusu9Df7LaKioB"
MODEL_PATH = "bestt_model.h5"

# ===============================
# 📥 Téléchargement du modèle
# ===============================
def download_model_from_drive(file_id: str, output_path: str):
    """Télécharge le modèle .h5 depuis Google Drive si absent"""
    if not os.path.exists(output_path):
        st.info("📥 Téléchargement du modèle depuis Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        st.success("✅ Téléchargement terminé avec succès !")
    else:
        st.info("✅ Modèle téléchargé.")

download_model_from_drive(MODEL_FILE_ID, MODEL_PATH)

# ===============================
# 🎨 Classes et palette de couleurs
# ===============================
class_names = ["plat", "humain", "véhicule", "construction", "objet", "nature", "ciel", "vide"]

palette_colors = [
    [128, 64, 128],   # plat
    [244, 35, 232],   # humain
    [70, 70, 70],     # véhicule
    [102, 102, 156],  # construction
    [190, 153, 153],  # objet
    [152, 251, 152],  # nature
    [70, 130, 180],   # ciel
    [0, 0, 0]         # vide
]


palette = np.array(palette_colors, dtype=np.uint8)

# ===============================
# 🧮 Fonctions de perte et métriques
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
# 🚀 Chargement du modèle (mise en cache)
# ===============================
@st.cache_resource
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

model = load_segmentation_model()

# ===============================
# 🎛️ Barre latérale
# ===============================
st.sidebar.header("⚙️ Segmentation ")
input_image = st.sidebar.file_uploader("📷 Image originale", type=["jpg", "png", "jpeg"])
mask_true = st.sidebar.file_uploader("🎭 Masque réel", type=["jpg", "png", "jpeg"])
run_button = st.sidebar.button("🔮 Lancer la segmentation")


# ===============================
# 🖼️ Zone principale : affichage
# ===============================
if input_image is not None:
    image = Image.open(input_image).convert("RGB")
    st.subheader("📷 Image originale")
    st.image(image, caption="Image originale", use_container_width=True)

    if run_button:
        with st.spinner("⏳ Prédiction en cours..."):
            img_resized = image.resize(IMG_SIZE[::-1])
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_batch)
            pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
            pred_color = palette[pred_mask]

        st.subheader(" Résultats de la segmentation")
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(image, caption="Image originale", use_container_width=True)
        with colB:
            if mask_true is not None:
                mask_img = Image.open(mask_true).resize(IMG_SIZE[::-1])
                st.image(mask_img, caption="Masque réel", use_container_width=True)
            else:
                st.warning("⚠️ Aucun masque réel fourni.")
        with colC:
            st.image(pred_color, caption="Masque prédit (colorisé)", use_container_width=True)
    else:
        st.info("⬅️ **Lancer la segmentation**.")
else:
    st.info("⬅️ Veuillez télécharger une image.")
