import pandas as pd
import pickle
from src.segment import preprocess_data, segment_customer
from sklearn.cluster import KMeans
import os

# Load data
data = pd.read_csv("data/online_retail.csv")

# Preprocess data to get RFM features and scaled data
rfm, rfm_scaled, scaler = preprocess_data(data)

# Build KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(rfm_scaled)

# Assign segments
rfm['Segment'] = kmeans.labels_

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/rfm_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("models/rfm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Models saved to 'models/rfm_model.pkl' and 'models/rfm_scaler.pkl'")
