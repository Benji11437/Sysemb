import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import segmentation_models as sm
import gdown

# ===============================
# Configuration Streamlit
# ===============================
st.set_page_config(page_title="Segmentation d'image", layout="wide")
st.title(" Application de Segmentation d'Images")

# ===============================
# Paramètres du modèle
# ===============================
BACKBONE = 'resnet50'
IMG_SIZE = (256, 512)
NUM_CLASSES = 8
# MODEL_FILE_ID = "1vjg08BuQTt1nc_eMLLBusu9Df7LaKioB"
MODEL_FILE_ID = "1SWEk8PpN_oM-jIAQ4Sy1-C2pLbCDfvYV"
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
        st.info("✅ Modèle déjà disponible.")

download_model_from_drive(MODEL_FILE_ID, MODEL_PATH)

# ===============================
# Classes et palette de couleurs
# ===============================
class_names = ["plat", "humain", "véhicule", "construction",
               "objet", "nature", "ciel", "vide"]

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
# Charger le modèle
# ===============================
@st.cache_resource
def load_model():
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

model = load_model()

# ===============================
# Liste des images disponibles
# ===============================
IMAGE_DIR = "images"
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_leftImg8bit.png")]

st.sidebar.header("📁 Sélection d'image")
selected_image = st.sidebar.selectbox("Choisissez une image dans la liste :", ["(Aucune)"] + images)
run_button = st.sidebar.button("Lancer la segmentation")

# ===============================
# Affichage principal
# ===============================
if selected_image == "(Aucune)":
    # ---- Message de bienvenue avec bannière ----
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://plus.unsplash.com/premium_photo-1676637656166-cb7b3a43b81a?fm=jpg&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YWklMjB0ZWNobm9sb2d5fGVufDB8fDB8fHww&ixlib=rb-4.1.0&q=60&w=3000"
                 alt="banner"
                 style="width:80%;border-radius:15px;margin-bottom:25px;box-shadow:0 4px 10px rgba(0,0,0,0.2);">            
            <h2> Bienvenue dans l’application de segmentation d’images</h2>
            
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # Une image a été sélectionnée
    base_name = selected_image.replace("_leftImg8bit.png", "")
    img_path = os.path.join(IMAGE_DIR, selected_image)
    mask_path = os.path.join(IMAGE_DIR, f"{base_name}_gtFine_labelIds.png")

    image = Image.open(img_path).convert("RGB")
    mask_real = Image.open(mask_path).convert("RGB") if os.path.exists(mask_path) else None

    st.subheader(f"🖼️ Image sélectionnée : {selected_image}")
    st.image(image, caption="Image originale", use_container_width=True)

    # Si le bouton est cliqué → lancer la segmentation
    if run_button:
        with st.spinner("🧠 Prédiction en cours..."):
            img_resized = image.resize(IMG_SIZE[::-1])
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_batch)
            pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
            pred_color = palette[pred_mask]

        # Affichage des résultats
        st.subheader("Résultat de la segmentation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Image originale", use_container_width=True)
        with col2:
            if mask_real:
                st.image(mask_real, caption="Masque réel", use_container_width=True)
            else:
                st.warning("Aucun masque réel trouvé pour cette image.")
        with col3:
            st.image(pred_color, caption="Masque prédit (colorisé)", use_container_width=True)
