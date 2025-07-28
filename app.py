import streamlit as st
import pandas as pd
import pickle
from src.segment import preprocess_data, segment_customer
import joblib
import numpy as np

# Load trained models
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("models/rfm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/rfm_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found in the 'models' directory.")
        return None, None

model, scaler = load_model_and_scaler()

st.title("🛍️ Shopper Spectrum: Customer Segmentation App")

st.markdown("Enter customer data to predict their segment based on RFM analysis.")

# Input fields
recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (number of purchases)", min_value=0)
monetary = st.number_input("Monetary Value (total spent)", min_value=0.0)

if st.button("Predict Segment"):
    try:
        # Load models
        kmeans = joblib.load("models/rfm_model.pkl")
        scaler = joblib.load("models/rfm_scaler.pkl")

        # Prepare input and scale
        user_input = np.array([[recency, frequency, monetary]])
        user_scaled = scaler.transform(user_input)

        # Predict segment
        segment = kmeans.predict(user_scaled)[0]

        # Define segment labels
        segment_labels = {
            0: "⚠️ At-Risk Customers",
            1: "🛍️ Occasional Buyers",
            2: "🏆 Loyal Spenders",
            3: "🧲 New or Potential Customers"
        }

        # Show result
        st.success(f"The predicted customer segment is: {segment_labels.get(segment, 'Unknown')}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
# --- Product Recommendation Section ---
st.markdown("---")
st.subheader("🛒 Product Recommendations Based on Similar Customers")

# Load recommender model
import joblib
from src.recommender import recommend_products

recommender_model = joblib.load("models/recommender.pkl")

# Dummy past product input (can be made interactive later)
past_product = st.text_input("Enter a product you purchased recently:", "WHITE HANGING HEART T-LIGHT HOLDER")

if st.button("Get Recommendations"):
    try:
        recommendations = recommend_products(recommender_model, past_product)
        if recommendations:
            st.success("✅ Recommended Products:")
            for i, product in enumerate(recommendations, 1):
                st.markdown(f"{i}. {product}")
        else:
            st.warning("⚠️ No similar products found. Try another input.")
    except Exception as e:
        st.error(f"⚠️ Error during recommendation: {e}")
