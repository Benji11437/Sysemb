import streamlit as st
import requests
import os
from PIL import Image
import io

# ===============================
# Configuration
# ===============================
FLASK_URL = "http://localhost:5000/segment"  # API Flask 


IMAGE_DIR = "images"
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_leftImg8bit.png")]

st.sidebar.header("📁 Sélection d'image")
selected_image = st.sidebar.selectbox("Choisissez une image :", ["(Aucune)"] + images)
run_button = st.sidebar.button("Lancer la segmentation")

# ===============================
# Affichage principal
# ===============================
if selected_image == "(Aucune)":
    st.title("Bienvenue 👋")
    st.write("Sélectionnez une image dans la barre latérale pour commencer.")
else:
    img_path = os.path.join(IMAGE_DIR, selected_image)
    image = Image.open(img_path).convert("RGB")
    
    st.subheader(f"🖼️ Image sélectionnée : {selected_image}")
    st.image(image, caption="Image originale", use_container_width=True)

    if run_button:
        with st.spinner("🧠 Envoi à Flask + Segmentation..."):

            # Prépare l’image pour l’API Flask
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Envoi à Flask
            response = requests.post(
                FLASK_URL,
                files={"image": ("image.png", img_bytes, "image/png")}
            )

            if response.status_code == 200:
                segmented_img = Image.open(io.BytesIO(response.content))

                st.subheader("Résultat de la segmentation")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Image originale", use_container_width=True)
                with col2:
                    st.image(segmented_img, caption="Masque prédit (Flask)", use_container_width=True)

            else:
                st.error(f"Erreur API Flask : {response.text}")
