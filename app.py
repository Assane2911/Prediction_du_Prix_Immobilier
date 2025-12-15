import streamlit as st
import pandas as pd
import joblib
import os

# --- Charger le modèle ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modele_linear_multiple.joblib")
model = joblib.load(MODEL_PATH)

st.title("🏠 Prédiction du Prix Immobilier - Régression Linéaire Multiple")

# --- Entrée utilisateur pour chaque feature ---
MedInc = st.number_input("Revenu médian des ménages (MedInc)", value=3.0)
HouseAge = st.number_input("Âge moyen des maisons (HouseAge)", value=30)
AveRooms = st.number_input("Nombre moyen de pièces (AveRooms)", value=5.0)
AveBedrms = st.number_input("Nombre moyen de chambres (AveBedrms)", value=1.0)
Population = st.number_input("Population", value=1000)
AveOccup = st.number_input("Occupation moyenne (AveOccup)", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# --- Mettre les données dans un DataFrame ---
input_data = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})

# --- Bouton pour prédire ---
if st.button("Prédire le prix"):
    prediction = model.predict(input_data)[0]
    st.success(f"Le prix médian prédit est : {prediction:.2f}")
