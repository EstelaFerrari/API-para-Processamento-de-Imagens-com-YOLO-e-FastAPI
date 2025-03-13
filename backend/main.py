import cv2
import numpy as np
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from PIL import Image
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(
    title="API YOLO com FastAPI",
    description="API para processamento de imagens e detecção de objetos com YOLO, com personalização de borda e texto.",
    version="2.0"
)

# Rota para a raiz
@app.get("/")
async def root():
    return {"message": "Bem-vindo à API YOLO!"}

# Mapeamento dos modelos disponíveis para os seus respectivos caminhos de arquivo
modelFiles = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt"
}

# Variáveis globais para o modelo atual
currentModel = "yolov8n"
detector = YOLO(modelFiles[currentModel])

@app.post("/mudar_modelo/")
async def mudar_modelo(model_version: str = Query("yolov8n", enum=list(modelFiles.keys()))):
    global detector, currentModel
    if model_version != currentModel:
        detector = YOLO(modelFiles[model_version])
        currentModel = model_version
        return {"message": f"Modelo alterado para {model_version}"}
    return {"message": f"O modelo atual já é {model_version}"}

@app.post("/processar_imagem/")
async def processar_imagem(
    image_file: UploadFile = File(...),
    selected_model: str = Query("yolov8n", enum=list(modelFiles.keys())),
    min_confidence: float = Query(0.25, description="Confiança mínima para detecção")
):
    global detector, currentModel

    # Atualiza o modelo se uma versão diferente for solicitada
    if selected_model != currentModel:
        detector = YOLO(modelFiles[selected_model])
        currentModel = selected_model

    # Lê a imagem e converte para o formato OpenCV
    image_data = await image_file.read()
    pilImage = Image.open(io.BytesIO(image_data))
    cvImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    # Realiza a detecção de objetos
    detectionResults = detector(cvImage)

    # Percorre as detecções e anota a imagem
    for result in detectionResults:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf >= min_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{detector.names[int(box.cls[0])]} {conf:.2f}"
                
                # Desenha a caixa delimitadora
                cv2.rectangle(cvImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cvImage, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Codifica a imagem anotada como JPEG e a retorna
    success, encodedImage = cv2.imencode(".jpg", cvImage)
    if not success:
        return {"error": "Falha ao codificar a imagem."}
    return StreamingResponse(io.BytesIO(encodedImage.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
