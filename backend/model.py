import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    def __init__(self):
        self.user_product_matrix = None
        self.user_similarity = None
        self.user_ids = None
        self.product_ids = None
        self.product_names = {}
        
    def load_and_train(self, csv_path):
        """Load dataset and train the recommendation model"""
        print("Loading dataset...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Use first 2000 rows for speed (change to 5000 if your computer can handle)
        df = df.head(2000)
        
        # Create product name mapping (some datasets have product names)
        if 'ProductId' in df.columns:
            self.product_ids = df['ProductId'].unique()
            # Create dummy product names if not present
            for idx, pid in enumerate(self.product_ids[:100]):
                self.product_names[pid] = f"Product_{idx+1}"
        
        # Create user-product rating matrix
        print("Creating user-product matrix...")
        self.user_product_matrix = df.pivot_table(
            index='UserId', 
            columns='ProductId', 
            values='Score',
            fill_value=0
        )
        
        # Calculate user similarity
        print("Calculating user similarities...")
        self.user_similarity = cosine_similarity(self.user_product_matrix)
        self.user_ids = self.user_product_matrix.index.tolist()
        
        print(f"✅ Model trained successfully!")
        print(f"   - Users: {len(self.user_ids)}")
        print(f"   - Products: {len(self.user_product_matrix.columns)}")
        
        return True
    
    def get_recommendations(self, user_id, num_recommendations=5):
        """Get product recommendations for a specific user"""
        
        # Check if user exists
        if user_id not in self.user_ids:
            return {"error": f"User '{user_id}' not found. Please try another user ID."}
        
        # Find similar users
        user_idx = self.user_ids.index(user_id)
        similar_users = self.user_similarity[user_idx].argsort()[::-1][1:6]
        
        # Get products liked by similar users
        recommendations = []
        recommendation_scores = []
        
        for sim_user in similar_users:
            sim_user_id = self.user_ids[sim_user]
            sim_user_products = self.user_product_matrix.loc[sim_user_id]
            
            # Get products with rating >= 4
            liked_products = sim_user_products[sim_user_products >= 4].index.tolist()
            
            for product in liked_products:
                if product not in recommendations:
                    recommendations.append(product)
                    # Get product name
                    product_name = self.product_names.get(product, f"Product_{product[:8]}")
                    recommendation_scores.append({
                        "product_id": product,
                        "product_name": product_name,
                        "confidence": round(self.user_similarity[user_idx][sim_user] * 100, 2)
                    })
        
        # Sort by confidence and return top N
        recommendation_scores.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendation_scores[:num_recommendations]
    
    def get_all_users(self, limit=10):
        """Get list of available users for the frontend dropdown"""
        return self.user_ids[:limit]

# Singleton instance
recommendation_engine = RecommendationEngine()