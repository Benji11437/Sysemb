# -----------------------------
# Étape 1 : Image de base
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# Étape 2 : Variables d'environnement
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------
# Étape 3 : Installer les dépendances système
# -----------------------------
RUN apt-get update && \
    apt-get install -y git gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Étape 4 : Copier et installer les dépendances Python
# -----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
# Installer TensorFlow CPU et les autres packages
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Étape 5 : Copier le code et le modèle
# -----------------------------
COPY . .

# -----------------------------
# Étape 6 : Exposer le port Streamlit
# -----------------------------
EXPOSE 8501

# -----------------------------
# Étape 7 : Commande de démarrage
# -----------------------------
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
