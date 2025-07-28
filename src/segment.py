# src/segment.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_data(df):
    df.dropna(subset=['CustomerID'], inplace=True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[(rfm['Monetary'] > 0)]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    return rfm, rfm_scaled, scaler

def segment_customer(recency, frequency, monetary, model, scaler):
    """
    Predict customer segment given Recency, Frequency, Monetary inputs.
    """
    rfm_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
    rfm_scaled = scaler.transform(rfm_df)
    segment = model.predict(rfm_scaled)[0]
    return segment


