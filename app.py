import streamlit as st
import pandas as pd
import pickle
from src.clustering import segment_customer
from src.recommender import recommend_products, pivot_table, similarity_matrix

# Load the trained KMeans model
kmeans_model = pickle.load(open("models/kmeans_model.pkl", "rb"))

# Load the customer-item matrix for product recommendation
customer_item_matrix = pickle.load(open("models/customer_item_matrix.pkl", "rb"))

# Define segment labels manually (based on your cluster analysis)
segment_map = {
    0: "Occasional Shoppers",
    1: "Loyal Customers",
    2: "Big Spenders",
    3: "Recent Low Spenders"
    # Modify if your model has a different number of clusters
}

# Set Streamlit page title
st.title("🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendations")

# Create two tabs
tab1, tab2 = st.tabs(["🧮 Customer Segmentation", "📦 Product Recommendation"])

# -------- Tab 1: Customer Segmentation --------
with tab1:
    st.header("Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, step=1.0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0.0, step=1.0)
    monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, step=1.0)

    if st.button("Predict Segment"):
        try:
            segment = segment_customer(recency, frequency, monetary, kmeans_model, segment_map)
            st.success(f"🧾 Customer belongs to Segment: **{segment}**")
        except Exception as e:
            st.error(f"⚠️ Error during segmentation: {e}")

# -------- Tab 2: Product Recommendation --------
with tab2:
    st.header("Product Recommendation System")

    product_name = st.text_input("Enter a product name (e.g., 'WHITE HANGING HEART T-LIGHT HOLDER')")

    if st.button("Recommend Similar Products"):
        if product_name:
            try:
                recommendations = recommend_products(product_name, pivot_table, similarity_matrix)
                st.write("📦 Recommended Products:")
                for i, prod in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {prod}")
            except KeyError:
                st.error("❌ Product not found. Please try a valid product name from the dataset.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")
        else:
            st.warning("⚠️ Please enter a product name to get recommendations.")
