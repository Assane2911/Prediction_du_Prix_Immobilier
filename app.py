# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="🏠 Prédiction du Prix Immobilier (California Housing)",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Prédiction du Prix Immobilier")
st.write("Entrez les caractéristiques de la maison pour obtenir une prédiction du prix.")
st.caption("*Le prix est prédit en utilisant le California Housing Dataset, avec des données de 1990.*")

MODEL_PATH = "modele_regression_lineaire.joblib"

# --- Chargement du modèle réel ---
if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ Fichier **{MODEL_PATH}** introuvable ! Assurez-vous qu'il est dans le dossier de l'app et qu'il contient le modèle entraîné sur l'ensemble de données California Housing.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Modèle chargé avec succès !")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")

# --- Inputs utilisateur ---
st.subheader("Caractéristiques de la maison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Rappel : MedInc est en dizaines de milliers de dollars (e.g., 5.0 = 50 000 $)
    MedInc = st.slider("Revenu médian (MedInc)", 0.0, 20.0, 5.0, 0.1)
    HouseAge = st.slider("Âge de la maison (HouseAge)", 0.0, 52.0, 20.0, 1.0)

with col2:
    AveRooms = st.slider("Nombre moyen de pièces (AveRooms)", 0.0, 50.0, 5.0, 0.1)
    AveBedrms = st.slider("Nombre moyen de chambres (AveBedrms)", 0.0, 5.0, 1.0, 0.1)

with col3:
    Population = st.slider("Population", 0, 3500, 1000, 10)
    AveOccup = st.slider("Occupation moyenne (AveOccup)", 0.0, 10.0, 3.0, 0.1)

with col4:
    # Latitude et Longitude sont cruciales pour la prédiction
    Latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.01)
    Longitude = st.slider("Longitude", -124.0, -114.0, -118.0, 0.01)

# --- Préparation des données pour la prédiction ---
X_input = pd.DataFrame([{
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}])

# --- Bouton de prédiction et Conversion ---
st.markdown("---")
if st.button("Prédire le prix 🏠"):
    if os.path.exists(MODEL_PATH):
        try:
            prediction_base = model.predict(X_input)[0]
            prediction_base = max(prediction_base, 0)  # Évite les valeurs négatives (dans l'unité du dataset)

            # --- LA CONVERSION EST ICI ---
            # Multiplier la prédiction par 100 000 pour obtenir le montant en dollars
            prix_en_dollars = prediction_base * 100000

            st.success(f"💰 Le prix médian prédit (en $100.000) est : **{prediction_base:,.2f}**")
            st.markdown(f"## 🏠 Prix estimé en dollars : **${prix_en_dollars:,.0f}**")
            st.info("Rappel : Ce modèle a été entraîné sur des données de 1990.")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
