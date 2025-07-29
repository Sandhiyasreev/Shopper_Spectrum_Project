# src/recommender.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("data/online_retail.csv", encoding='ISO-8859-1')
data.dropna(subset=['CustomerID', 'Description'], inplace=True)

# Create product-customer matrix
pivot_table = pd.pivot_table(
    data,
    index='Description',
    columns='CustomerID',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Compute cosine similarity between products
similarity_matrix = cosine_similarity(pivot_table)

# Create reverse mapping of product names to indices
product_index = pd.Series(pivot_table.index)


# Recommendation function
def recommend_products(product_name, pivot_table, similarity_matrix, top_n=5):
    if product_name not in pivot_table.index:
        raise KeyError("Product not found in dataset.")

    index = pivot_table.index.get_loc(product_name)
    similarity_scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    recommended_products = [pivot_table.index[i] for i, score in sorted_scores]
    return recommended_products
