# src/segmenter.py

import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/rfm_kmeans_model.pkl")
scaler = joblib.load("models/rfm_scaler.pkl")

# Map cluster number to human-readable label
cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}


def predict_customer_segment(recency, frequency, monetary):
    # Create feature vector
    input_data = np.array([[recency, frequency, monetary]])

    # Scale the input using the trained scaler
    scaled_data = scaler.transform(input_data)

    # Predict the cluster
    cluster = model.predict(scaled_data)[0]

    # Return the mapped cluster label
    return cluster_labels.get(cluster, "Unknown")
