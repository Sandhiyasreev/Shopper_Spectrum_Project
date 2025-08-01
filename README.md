# 🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendation 🧠📦🔍

A full-fledged Machine Learning project that segments e-commerce customers based on RFM (Recency, Frequency, Monetary) behavior and recommends products using item-based collaborative filtering. Built to empower online retailers with intelligent insights and tailored customer experiences.

# 🔧 Features

🧭 Customer Segmentation
📊 RFM Analysis – Calculates customer value based on:
Recency: How recently a customer made a purchase
Frequency: How often they purchase
Monetary: How much money they spend

# 📌 KMeans Clustering – Groups customers into segments:

Loyal
Potential Loyalist
At Risk
Occasional

🛍️ Helps target specific customer groups for personalized marketing
🎯 Product Recommendation
🧠 Item-Based Collaborative Filtering
🧮 Uses Cosine Similarity on purchase patterns
🔄 Recommends similar products based on past purchases
Boosts customer engagement and repeat purchases

# 🧠 Technologies Used

Python (Pandas, NumPy, Scikit-learn)
Machine Learning
KMeans Clustering (Segmentation)
Cosine Similarity (Recommendation)
Data Preprocessing (handling duplicates, nulls, outliers)
Streamlit (for UI – optional)
Jupyter Notebook (for model development)

# 📁 Project Structure

Shopper_Spectrum_Project/
├── app.py                               # Streamlit App (UI)
├── src/
│   ├── clustering.py                    # RFM + KMeans logic
│   └── recommender.py                   # Recommendation engine
├── models/
│   ├── kmeans_model.pkl                 # Saved KMeans model
│   └── customer_item_matrix.pkl         # Precomputed item-item similarity
├── data/
│   └── online_retail.csv                # Transaction data
├── Mohammed_Salman_Shopper_Spectrum_Project.ipynb  # Final notebook
├── README.md                            # Project description
└── requirements.txt                     # Python dependencies

# 🧪 Example Use Cases

📌 Loyal Customers → Send premium offers
📌 At-Risk Customers → Re-engagement campaign
📌 Similar Product Suggestions → Upselling and cross-selling opportunities
📌 Customer Insights → Data-driven marketing and segmentation

## 🙋‍♀️ Created By

**Sandhiya Sree V**  
📧 sandhiyasreev@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sandhiya-sree-v-3a2321298/)  
🌐 [Github](https://github.com/Sandhiyasreev)

# 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share with credit.

⭐ If you found this project helpful, give it a star!
💬 For feedback or collaboration, feel free to reach out.
