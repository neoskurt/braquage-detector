import streamlit as st
import requests
import cv2
import base64
import numpy as np
import time
import pygame

# Charger le détecteur de visage MTCNN
from mtcnn import MTCNN
detector = MTCNN()

# Initialiser Pygame pour jouer des sons
pygame.mixer.init()

# Charger le son
shots_sound = pygame.mixer.Sound('shots.mp3')

# Variables pour suivre la détection du braqueur
detected_braqueur = False
start_time = None
alert_triggered = False

# Seuil de confiance pour considérer un braqueur
CONFIDENCE_THRESHOLD = 0.50

def capture_and_predict():
    global detected_braqueur, start_time, alert_triggered
    
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

        braqueur_detected_in_frame = False

        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', face_img)
            image_base64 = base64.b64encode(img_encoded).decode('utf-8')

            try:
                response = requests.post("http://localhost:8000/predict", json={"image": image_base64})
                response.raise_for_status()  # Cette ligne déclenchera une exception si la requête échoue
                prediction = response.json()["prediction"]
                confidence = prediction['confidence']
                if confidence > CONFIDENCE_THRESHOLD:
                    color = (0, 0, 255)  # Rouge pour les braqueurs
                    braqueur_detected_in_frame = True
                    prediction_text.markdown(f"**Prédiction : Braqueur** **Confiance :** {confidence:.2f}")
                else:
                    color = (0, 255, 0)  # Vert pour les non-braqueurs
                    prediction_text.markdown(f"**Prédiction : Non Braqueur** **Confiance :** {confidence:.2f}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            except requests.exceptions.RequestException as e:
                prediction_text.markdown(f"Erreur lors de la prédiction : {e}")

        # Logic to check detection duration
        if braqueur_detected_in_frame:
            if not detected_braqueur:
                detected_braqueur = True
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time > 6 and not alert_triggered:
                    shots_sound.play()
                    alert_triggered = True
        else:
            detected_braqueur = False
            start_time = None
            alert_triggered = False

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
