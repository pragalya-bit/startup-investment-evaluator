import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import os

# ==============================
# DOWNLOAD MODEL FROM DRIVE
# ==============================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ggTEaM2FxFOTar-YQ8-mzuSuZMCfOhtN"
MODEL_PATH = "high_growth_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading pre-trained model...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("Model downloaded!")

# Load the model
model = joblib.load(MODEL_PATH)

# ==============================
# Streamlit App UI
# ==============================
st.set_page_config(page_title="AI Investment Scoring Engine", layout="centered")
st.title("🚀 AI Startup Investment Analyzer")
st.markdown("Predict if a startup idea is likely **high-growth** or risky.")

st.divider()

# User inputs
category = st.text_input("Startup Category (e.g., AI, FinTech, Health)")
market = st.text_input("Market Type")
country = st.text_input("Country Code (e.g., USA, IND)")
state = st.text_input("State Code (e.g., CA, NY)")
funding_rounds = st.number_input("Expected Funding Rounds", 0, 20, 1)
founded_year = st.number_input("Founded Year", 1990, 2026, 2024)

st.divider()

if st.button("🔍 Analyze Investment Potential"):
    input_data = pd.DataFrame({
        "category_list": [category],
        "market": [market],
        "country_code": [country],
        "state_code": [state],
        "funding_rounds": [funding_rounds],
        "founded_year": [founded_year]
    })

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]
    score = round(probability * 100, 2)

    # Display score
    if score >= 80:
        st.success(f"Investment Score: {score}% – 🟢 STRONG INVESTMENT")
    elif score >= 60:
        st.warning(f"Investment Score: {score}% – 🟡 MODERATE POTENTIAL")
    else:
        st.error(f"Investment Score: {score}% – 🔴 HIGH RISK")

    # Progress bar
    st.progress(int(score))

    # Probability chart
    st.subheader("📈 Growth Probability")
    labels = ["Low Growth", "High Growth"]
    values = [1 - probability, probability]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["red", "green"])
    ax.set_ylim([0, 1])
    st.pyplot(fig)

st.divider()
st.caption("Final Year Project - AI-Based Startup Growth Prediction System")
