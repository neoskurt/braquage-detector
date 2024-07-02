import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

model_path = "model/saved_model.h5"

def preprocess_image(image: str):
    img_data = base64.b64decode(image)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

async def train_model():
    images = []
    labels = []

    # Dossier contenant les images de braqueur
    braqueur_dir = "img/train/braqueur"
    for file_name in os.listdir(braqueur_dir):
        file_path = os.path.join(braqueur_dir, file_name)
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img_to_array(img))
            labels.append(1)  # Étiquette 1 pour braqueur

    # Dossier contenant les images de non-braqueur
    non_braqueur_dir = "img/train/non-braqueur"
    for file_name in os.listdir(non_braqueur_dir):
        file_path = os.path.join(non_braqueur_dir, file_name)
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img_to_array(img))
            labels.append(0)  # Étiquette 0 pour non-braqueur

    images = np.array(images)
    labels = np.array(labels)
    images /= 255.0

    # Définition du modèle
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Couche de sortie pour la classification binaire
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(images, labels, epochs=10, validation_split=0.2)

    model.save(model_path)

async def predict_image(image: str, threshold=0.5):
    img = preprocess_image(image)
    model = load_model(model_path)
    prediction = model.predict(img)
    confidence = prediction[0][0]  # Obtenir la probabilité de la prédiction
    predicted_class = (confidence > threshold).astype(int)  # Convertir la sortie en classe (0 ou 1)

    if predicted_class == 1:
        result = "Braqueur"
    else:
        result = "Non Braqueur"
    
    return {
        "result": result,
        "confidence": float(confidence)
    }
