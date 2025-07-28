import pandas as pd

# Load the dataset (adjust path if needed)
df = pd.read_csv("data/online_retail.csv", encoding='ISO-8859-1')

# Display unique product names
product_names = df['Description'].dropna().unique()

# Display first 20 product names
for i, name in enumerate(product_names[:20], 1):
    print(f"{i}. {name}")
