# src/model.py

import numpy as np


def segment_customer(rfm_scaled, model, scaler, recency, frequency, monetary):
    try:
        # Scale the input
        input_data = scaler.transform([[recency, frequency, monetary]])

        # Predict cluster
        cluster = model.predict(input_data)[0]

        # Return cluster number
        return cluster
    except Exception as e:
        print(f"⚠️ Error in segment_customer: {e}")
        return None
