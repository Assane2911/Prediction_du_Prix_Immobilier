# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- Configuration de la page ---
st.set_page_config(
    page_title="🏠 Prédiction du Prix Immobilier",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Prédiction du Prix Immobilier")
st.write("Entrez les caractéristiques de la maison pour obtenir une prédiction du prix.")

MODEL_PATH = "modele_regression_lineaire.joblib"

# --- Fonction pour créer un pipeline si le fichier est absent ou incompatible ---
def create_model(path):
    st.warning("⚠️ Modèle introuvable ou incompatible. Création d'un pipeline par défaut...")
    X_train = pd.DataFrame([[1,2,3,4,5,6,7,8]], columns=[
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"])
    y_train = [100000]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, path)
    st.success("✅ Modèle créé et sauvegardé !")
    return pipeline

# --- Chargement du modèle avec gestion des erreurs ---
try:
    if not os.path.exists(MODEL_PATH):
        model = create_model(MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Modèle chargé avec succès !")
except Exception:
    # Si erreur lors du chargement (incompatibilité), recrée le modèle
    model = create_model(MODEL_PATH)

# --- Inputs utilisateur avec sliders et layout en colonnes ---
st.subheader("Caractéristiques de la maison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    medinc = st.slider("Revenu médian (MedInc)", 0.0, 20.0, 5.0, 0.1)
    house_age = st.slider("Âge de la maison (HouseAge)", 0.0, 100.0, 20.0, 1.0)

with col2:
    ave_rooms = st.slider("Nombre moyen de pièces (AveRooms)", 0.0, 20.0, 5.0, 0.1)
    ave_bedrms = st.slider("Nombre moyen de chambres (AveBedrms)", 0.0, 10.0, 1.0, 0.1)

with col3:
    population = st.slider("Population", 0, 5000, 1000, 10)
    ave_occup = st.slider("Occupation moyenne (AveOccup)", 0.0, 10.0, 3.0, 0.1)

with col4:
    latitude = st.slider("Latitude", -90.0, 90.0, 34.0, 0.01)
    longitude = st.slider("Longitude", -180.0, 180.0, -118.0, 0.01)

# --- Préparation des données pour la prédiction ---
X = pd.DataFrame([{
    "MedInc": medinc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude
}])

# --- Bouton de prédiction ---
st.markdown("---")
if st.button("Prédire le prix 🏠"):
    try:
        prediction = model.predict(X)
        st.success(f"💰 Le prix prédit de la maison est : **{prediction[0]:,.2f} $**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
