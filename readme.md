
# Projet de Classification d'Images

Ce projet utilise FastAPI pour créer une API de détection de braquage et Streamlit pour créer une interface utilisateur. Le modèle est entraîné pour reconnaître les images de braqueurs et de non-braqueurs en temps réel.

Par

Sabatier Archibald /
Meziane Anis /
Truong Hubert

## Prérequis

- Python 3.8 ou supérieur
- `pip` pour installer les dépendances Python

## Installation

1. Clonez le dépôt du projet :

   ```bash
   git clone <URL_DU_DEPOT>
   cd projet-ia-efrei
   ```

2. Créez un environnement virtuel :

   ```bash
   python -m venv venv
   ```

3. Activez l'environnement virtuel :

   - Sur macOS et Linux :

     ```bash
     source venv/bin/activate
     ```

   - Sur Windows :

     ```bash
     venv\Scripts\activate.bat
     ```

4. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

## Lancer le projet

### Terminal 1 : Lancer le serveur FastAPI

1. Assurez-vous que l'environnement virtuel est activé.
2. Démarrez le serveur FastAPI avec la commande suivante :

   ```bash
   uvicorn api:app --reload
   ```

   Cette commande démarre le serveur FastAPI et il sera accessible à l'adresse `http://127.0.0.1:8000`.

### Terminal 2 : Lancer l'application Streamlit

1. Ouvrez un deuxième terminal.
2. Assurez-vous que l'environnement virtuel est activé dans ce terminal également.
3. Lancez l'application Streamlit avec la commande suivante :

   ```bash
   streamlit run app.py
   ```

   Cette commande démarre l'application Streamlit et elle sera accessible à l'adresse `http://localhost:8501`.

## Utilisation

### Entraîner le modèle

Pour entraîner le modèle, envoyez une requête POST à l'endpoint `/training` :

```bash
curl -X POST "http://127.0.0.1:8000/training"
```


### Faire une prédiction

Pour faire une prédiction avec une image, envoyez une requête POST à l'endpoint `/predict` avec l'image :

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path/to/your/image.jpg"
```

### Afficher les statistiques

Pour afficher les statistiques du modèle, envoyez une requête GET à l'endpoint `/stats` :

```bash
curl -X GET "http://127.0.0.1:8000/stats"
```

### Afficher les performances

Pour afficher les performances du modèle, envoyez une requête GET à l'endpoint `/performance` :

```bash
curl -X GET "http://127.0.0.1:8000/performance"
```

### Afficher le temps de traitement moyen

Pour afficher le temps de traitement moyen par image, envoyez une requête GET à l'endpoint `/time` :

```bash
curl -X GET "http://127.0.0.1:8000/time"
```

ou allez directement dans le navigateur

### Interagir avec l'application Streamlit

Utilisez l'interface Streamlit pour télécharger des images pour prédiction et télécharger le modèle entraîné. Ouvrez votre navigateur et accédez à `http://localhost:8501`.
