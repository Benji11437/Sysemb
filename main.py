import streamlit as st
import requests
from PIL import Image
import io
import os

# ===============================
# Configuration
# ===============================
FLASK_URL = "http://localhost:5000/segment"  # URL de l'API Flask  
IMAGE_DIR = "images"  # dossier contenant des images de test (optionnel)

# ===============================
# Barre latérale : sélection ou upload
# ===============================
st.sidebar.header("📁 Sélection d'image")

# Liste des images locales disponibles
images = []
if os.path.exists(IMAGE_DIR):
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

selected_image = st.sidebar.selectbox(
    "Choisissez une image existante :", ["(Aucune)"] + images
)

uploaded_file = st.sidebar.file_uploader("Ou téléversez votre image :", type=["png", "jpg", "jpeg"])

run_button = st.sidebar.button("Lancer la segmentation")

# ===============================
# Fonction utilitaire pour envoyer l'image à Flask
# ===============================
def send_to_flask(img: Image.Image) -> Image.Image:
    """Envoie l'image à l'API Flask et récupère le masque colorisé"""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"image": ("image.png", img_bytes, "image/png")}
    response = requests.post(FLASK_URL, files=files)

    if response.status_code == 200:
        mask_img = Image.open(io.BytesIO(response.content))
        return mask_img
    else:
        st.error(f"Erreur lors de la segmentation : {response.status_code}")
        return None

# ===============================
# Affichage principal
# ===============================
if selected_image == "(Aucune)" and uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center;">
            <h2>Bienvenue dans l’application de segmentation d’images</h2>
            <p>Veuillez uploader une image ou en sélectionner une existante dans la barre latérale.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # Charger l'image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image_path = os.path.join(IMAGE_DIR, selected_image)
        image = Image.open(image_path).convert("RGB")

    st.subheader("🖼️ Image originale")
    st.image(image, use_container_width=True)

    # Bouton pour lancer la segmentation
    if run_button:
        with st.spinner("🧠 Segmentation en cours..."):
            mask_color = send_to_flask(image)

        if mask_color:
            st.subheader("🎨 Masque segmenté")
            st.image(mask_color, use_container_width=True)


