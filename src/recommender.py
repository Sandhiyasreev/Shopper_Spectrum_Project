import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_item_similarity_matrix(df):
    """
    Builds an item-item similarity matrix using cosine similarity.
    Args:
        df (pd.DataFrame): The transactional data in pivot table format (customers x products).
    Returns:
        pd.DataFrame: Item similarity matrix (products x products).
    """
    # Transpose for item-based similarity
    item_matrix = df.T
    similarity = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(similarity,
                                 index=item_matrix.index,
                                 columns=item_matrix.index)

    return similarity_df


def recommend_products(model, product_name, top_n=5):
    """
    Recommends top N products similar to the input product based on the similarity matrix.
    Args:
        model (pd.DataFrame): Precomputed item similarity matrix.
        product_name (str): Name of the product for which recommendations are needed.
        top_n (int): Number of recommendations to return.
    Returns:
        List[str]: Recommended product names.
    """
    if product_name not in model.columns:
        return []

    # Sort similar products in descending order of similarity score
    scores = model[product_name].sort_values(ascending=False)

    # Remove the input product itself and return top N
    recommended_products = scores.drop(product_name).head(top_n).index.tolist()

    return recommended_products
