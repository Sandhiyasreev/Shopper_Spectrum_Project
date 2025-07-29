import pandas as pd
import numpy as np

def preprocess_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df.dropna(inplace=True)
    df = df[df['InvoiceNo'].astype(str).str.startswith('C') == False]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    latest_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return df, rfm
