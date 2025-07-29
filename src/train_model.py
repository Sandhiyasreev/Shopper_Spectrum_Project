from src.preprocessing import preprocess_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

df, rfm = preprocess_data("data/online_retail.csv")

# Scale RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(rfm_scaled)

# Save KMeans model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Build customer-item matrix
customer_item_matrix = pd.pivot_table(
    df, index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0
)

# Save customer-item matrix
with open("models/customer_item_matrix.pkl", "wb") as f:
    pickle.dump(customer_item_matrix, f)

print("✅ Training complete. Models saved.")
