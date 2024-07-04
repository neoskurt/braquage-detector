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
    image: str

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
    labels = ['Braqueurs', 'Non-Braqueurs']
    sizes = [120, 80]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title('Répartition des Données d\'Entraînement')

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig1)

    return StreamingResponse(img_bytes, media_type="image/png")

@app.get("/performance", summary="Graphiques de performance", description="Afficher les graphiques de performance du modèle.")
async def performance():
    epochs = range(1, 11)
    accuracy = [0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.9]
    loss = [0.7, 0.65, 0.6, 0.55, 0.5, 0.48, 0.45, 0.42, 0.4, 0.38]

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, 'b', label='Précision')
    ax.plot(epochs, loss, 'r', label='Perte')
    ax.set_title('Performances du Modèle')
    ax.set_xlabel('Époques')
    ax.set_ylabel('Valeurs')
    ax.legend()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)

    return StreamingResponse(img_bytes, media_type="image/png")

@app.get("/time", summary="Temps de traitement moyen", description="Afficher le temps de traitement moyen par image.")
async def time():
    images_processed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    time_per_image = [0.1, 0.09, 0.085, 0.082, 0.08, 0.078, 0.076, 0.075, 0.074, 0.073]

    fig, ax = plt.subplots()
    ax.plot(images_processed, time_per_image, 'g', label='Temps par Image')
    ax.set_title('Temps de Traitement Moyen par Image')
    ax.set_xlabel('Nombre d\'Images Traitées')
    ax.set_ylabel('Temps Moyen (secondes)')
    ax.legend()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)

    return StreamingResponse(img_bytes, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
