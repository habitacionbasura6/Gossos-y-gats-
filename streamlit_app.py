import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Función para descargar el archivo desde una URL (si es necesario)
def download_file(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

# Verificar si los archivos existen y descargarlos si no es así
if not os.path.exists("model_gats_gossos.json") or not os.path.exists("model_gats_gossos.weights.h5"):
    st.info("⚠️ Descargando los archivos del modelo desde GitHub...")
    # Reemplaza con las URLs correctas de tus archivos en GitHub
    json_url = 'https://github.com/habitacionbasura6/Gossos-y-gats-/blob/main/model_gats_gossos.json'  # Reemplaza esta URL
    weights_url = 'https://github.com/habitacionbasura6/Gossos-y-gats-/blob/main/model_gats_gossos.weights.h5'  # Reemplaza esta URL
   
    # Descargar los archivos
    download_file(json_url, 'model_gats_gossos.json')
    download_file(weights_url, 'model_gats_gossos.weights.h5')

# Cargar el modelo
with open("model_gats_gossos.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("model_gats_gossos.weights.h5")

# Configuración de la página
st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("🐶 Classificador de Gossos i Gats 🐱")
st.markdown("Puja una imatge i la IA et dirà si veu un gos o un gat! 🧠")

# Subir imagen
uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Preprocesar la imagen
        image = Image.open(uploaded_file).convert("RGB").resize((100, 100))
        st.image(image, caption='📷 Imatge pujada', use_container_width=True)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Hacer la predicción
        prediction = model.predict(img_array)
        prob = float(prediction[0])

        # Mostrar el resultado
        if prob > 0.5:
            st.success(f"És un **gos** 🐶 amb {prob*100:.2f}% de confiança!")
        else:
            st.success(f"És un **gat** 🐱 amb {(1 - prob)*100:.2f}% de confiança!")

    except UnidentifiedImageError:
        st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
