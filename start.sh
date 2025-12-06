#!/bin/bash
set -e

# -------------------------
# Variables
# -------------------------
VENV_NAME="antenv"
APP_DIR="/home/site/wwwroot"

# -------------------------
# Création de l'environnement virtuel si absent
# -------------------------
if [ ! -d "$APP_DIR/$VENV_NAME" ]; then
    python3 -m venv $APP_DIR/$VENV_NAME
fi

# -------------------------
# Activation de l'environnement virtuel
# -------------------------
source $APP_DIR/$VENV_NAME/bin/activate

# -------------------------
# Mise à jour pip + dépendances
# -------------------------
pip install --upgrade pip setuptools wheel
pip install -r $APP_DIR/requirements.txt

# -------------------------
# Lancer Flask (Gunicorn) en arrière-plan
# -------------------------
echo "Starting Flask (Gunicorn)..."
gunicorn -b 0.0.0.0:5000 server:app --timeout 600 --workers 1 --threads 2 &

# -------------------------
# Lancer Streamlit (port 80)
# -------------------------
echo "Starting Streamlit..."
exec streamlit run $APP_DIR/main.py --server.port 80 --server.address 0.0.0.0
