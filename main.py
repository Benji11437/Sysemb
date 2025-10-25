import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import segmentation_models as sm

# ===============================
# Configuration Streamlit
# ===============================
st.set_page_config(page_title="Segmentation d'image", layout="wide")
st.title("Application de Segmentation d'Image")

# ===============================
# Paramètres du modèle
# ===============================
BACKBONE = 'resnet50'
IMG_SIZE = (256, 512)
NUM_CLASSES = 8

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

    # ⚠️ Remplace le chemin ci-dessous par celui de ton modèle
    model.load_weights("bestt_model.h5")
    return model

model = load_model()


# ===============================
# 🎛️ Barre latérale (menu)
# ===============================
st.sidebar.header("⚙️ Segmentation ")
st.sidebar.write("Télécharger vos images ci-dessous :")

input_image = st.sidebar.file_uploader("📷 Image originale", type=["jpg", "png", "jpeg"])
mask_true = st.sidebar.file_uploader("🎭 Masque réel", type=["jpg", "png", "jpeg"])

run_button = st.sidebar.button("🔮 Lancer la segmentation")

st.sidebar.markdown("---")
st.sidebar.write("**Informations sur les classes :**")
for name, color in zip(class_names, palette_colors):
    st.sidebar.markdown(
        f'<span style="background-color: rgb({color[0]}, {color[1]}, {color[2]}); '
        f'display:inline-block; width:18px; height:18px; border-radius:3px; margin-right:8px;"></span> {name}',
        unsafe_allow_html=True
    )

# ===============================
# 🖼️ Zone principale : affichage
# ===============================
if input_image is not None:
    image = Image.open(input_image).convert("RGB")

    # ➤ Afficher directement l'image originale
    st.subheader("📷 Image originale ")
    st.image(image, caption="Image originale", width='stretch')

    # --- Bouton pour lancer la segmentation ---
    if run_button:
        with st.spinner("Prédiction en cours..."):
            # Prétraitement
            img_resized = image.resize(IMG_SIZE[::-1])
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Prédiction
            pred = model.predict(img_batch)
            pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

            # Appliquer la palette de couleurs
            pred_color = palette[pred_mask]

        # ===============================
        # Affichage des résultats
        # ===============================
        st.subheader("Résultats de la segmentation")
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(image, caption="Image originale", width='stretch')
        with colB:
            if mask_true is not None:
                mask_img = Image.open(mask_true).resize(IMG_SIZE[::-1])
                st.image(mask_img, caption="Masque réel", width='stretch')
            else:
                st.warning("Veuillez téléverser le masque réel.")
        with colC:
            st.image(pred_color, caption="Masque prédit (colorisé)", width='stretch')
    else:
        st.info("➡️ Cliquez sur **Lancer la segmentation** dans le menu à gauche pour continuer.")
else:
    st.info("⬅️ Veuillez téléverser une image dans le menu à gauche pour commencer.")
