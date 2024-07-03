from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from function import train_model, predict_image
import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

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

@app.get("/stats", summary="Afficher les statistiques", description="Afficher les statistiques du modèle.")
async def stats():
    # Générer les graphiques ici
    labels = ['Braqueurs', 'Non-Braqueurs']
    sizes = [120, 80]  # Exemple de nombre d'images

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title('Répartition des Données d\'Entraînement')

    # Sauvegarder le graphique dans un objet BytesIO
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig1)  # Fermer le graphique pour libérer de la mémoire

    return StreamingResponse(img_bytes, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
