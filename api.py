from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from function import train_model, predict_image

app = FastAPI(
    title="Détecteur de Braquage",
    description="API pour détecter les braquages à partir d'images et de la webcam en temps réel.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/training", summary="Entraîner le modèle", description="Entraîner un modèle de détection de braquage sur les images fournies.")
async def training():
    try:
        await train_model()
        return {"message": "Modèle entraîné avec succès"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", summary="Prédire une image", description="Faire une prédiction de braquage à partir d'une image.")
async def predict(request: PredictionRequest):
    try:
        prediction = await predict_image(request.image)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
