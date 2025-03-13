import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8080"

# Título da aplicação
st.title("Detecção de Objetos com YOLO 🚀")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# Configurações do Modelo
st.sidebar.header("Configurações da Detecção")
selected_model = st.sidebar.selectbox("Modelo YOLO", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
min_confidence = st.sidebar.slider("Confiança mínima", 0.0, 1.0, 0.25)

# Enviar a imagem para processamento
if uploaded_file:
    st.image(uploaded_file, caption="Imagem Original", use_container_width=True)

    # Botão para processar a imagem
    if st.button("Processar Imagem"):
        with st.spinner("Processando..."):
            # Enviar imagem para API
            files = {"image_file": uploaded_file}
            params = {
                "selected_model": selected_model,
                "min_confidence": min_confidence,
            }

            response = requests.post(f"{API_URL}/processar_imagem/", files=files, params=params)

            if response.status_code == 200:
                # Exibir a imagem processada
                processed_image = Image.open(io.BytesIO(response.content))
                st.image(processed_image, caption="Imagem Processada", use_container_width=True)
            else:
                st.error(f"Erro ao processar a imagem. Status: {response.status_code}.")
                st.text(response.text)
