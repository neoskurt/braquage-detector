import streamlit as st
import requests
import cv2
import base64
import numpy as np
from mtcnn import MTCNN

# Charger le détecteur de visage MTCNN
detector = MTCNN()

def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur lors de l'ouverture de la webcam")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)  # Ajustez la luminosité si nécessaire

    frame_window = st.image([])
    prediction_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture de l'image")
            break

        # Détecter les visages
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', face_img)
            image_base64 = base64.b64encode(img_encoded).decode('utf-8')

            try:
                response = requests.post("http://localhost:8000/predict", json={"image": image_base64})
                response.raise_for_status()  # Cette ligne déclenchera une exception si la requête échoue
                prediction = response.json()["prediction"]
                if prediction['result'] == "Braqueur":
                    color = (0, 0, 255)  # Rouge pour les braqueurs
                else:
                    color = (0, 255, 0)  # Vert pour les non-braqueurs
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                prediction_text.markdown(f"**Prédiction :** {prediction['result']} **Confiance :** {prediction['confidence']:.2f}")
            except requests.exceptions.RequestException as e:
                prediction_text.markdown(f"Erreur lors de la prédiction : {e}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

        # Ajout d'une pause pour permettre de voir les mises à jour
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

st.title("Détecteur de Braquage")
st.write("Entraînez le modèle ou utilisez la webcam pour une prédiction en temps réel.")

# Entraînement du modèle
st.header("Entraîner le modèle")
if st.button("Entraîner", key="train_button"):
    try:
        response = requests.post("http://localhost:8000/training")
        response.raise_for_status()  # Cette ligne déclenchera une exception si la requête échoue
        st.write(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'entraînement : {e}")

# Prédiction en temps réel
st.header("Prédiction en temps réel")
capture_and_predict()
