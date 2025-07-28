import pandas as pd
import joblib
from recommender import build_item_similarity_matrix

# Load cleaned data
df = pd.read_csv("data/online_retail.csv")

# Create basket matrix: rows = customers, columns = products, values = quantities
basket = df.pivot_table(index='CustomerID',
                        columns='Description',
                        values='Quantity',
                        aggfunc='sum',
                        fill_value=0)

# Build similarity matrix
similarity_matrix = build_item_similarity_matrix(basket)

# Save to file
joblib.dump(similarity_matrix, "models/recommender.pkl")

print("✅ Recommender model saved successfully to models/recommender.pkl")
