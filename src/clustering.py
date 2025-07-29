import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

def train_kmeans_model(rfm_df, n_clusters=4):
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(rfm_scaled)

    # Save clustered data (optional)
    rfm_df['Cluster'] = model.predict(rfm_scaled)
    rfm_df.to_csv("data/rfm_clusters.csv", index=False)

    return model, scaler

def save_model(model, scaler, model_path="models/best_kmeans_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, scaler), model_path)

def predict_cluster(rfm_values, model_path="models/best_kmeans_model.pkl"):
    import numpy as np
    model, scaler = joblib.load(model_path)
    values_scaled = scaler.transform([rfm_values])
    return model.predict(values_scaled)[0]

def segment_customer(recency, frequency, monetary, model, segment_map):
    input_data = np.array([[recency, frequency, monetary]])
    cluster = model.predict(input_data)[0]
    return segment_map.get(cluster, f"Unknown Segment (Cluster {cluster})")

