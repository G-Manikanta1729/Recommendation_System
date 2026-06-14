import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Try to import huggingface hub (will fail locally if not installed)
try:
    from huggingface_hub import hf_hub_download
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("⚠️ huggingface_hub not installed. Will use local dataset only.")

class RecommendationEngine:
    def __init__(self):
        self.user_product_matrix = None
        self.user_similarity = None
        self.user_ids = None
        self.product_ids = None
        self.product_names = {}
        
    def download_dataset_from_hub(self):
        """Download dataset from Hugging Face if not present locally"""
        
        # Check if dataset already exists locally
        if os.path.exists('Reviews.csv'):
            print("✅ Dataset found locally: Reviews.csv")
            return 'Reviews.csv'
        
        if os.path.exists('backend/Reviews.csv'):
            print("✅ Dataset found locally: backend/Reviews.csv")
            return 'backend/Reviews.csv'
        
        print("📥 Dataset not found locally. Downloading from Hugging Face...")
        print("   (This may take 2-3 minutes for the first download)")
        
        if not HUB_AVAILABLE:
            print("❌ huggingface_hub not available. Please install it: pip install huggingface-hub")
            return None
        
        try:
            # Download from your Hugging Face dataset
            dataset_path = hf_hub_download(
                repo_id="GManikanta1729/amazon-reviews-recommendation",
                filename="Reviews.csv",
                resume=True,
                etag_timeout=30
            )
            print(f"✅ Dataset downloaded successfully to: {dataset_path}")
            
            # Copy to current directory for easier access
            import shutil
            local_path = 'Reviews.csv'
            if dataset_path != local_path and not os.path.exists(local_path):
                shutil.copy2(dataset_path, local_path)
                print(f"   Copied to: {local_path}")
                return local_path
            
            return dataset_path
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            print("\n📌 Manual download option:")
            print("   Visit: https://huggingface.co/datasets/GManikanta1729/amazon-reviews-recommendation")
            print("   Click 'Download' button and place Reviews.csv in the backend folder")
            return None
        
    def load_and_train(self, csv_path=None):
        """Load dataset and train the recommendation model"""
        
        # If no path provided, try to download from Hugging Face
        if csv_path is None or not os.path.exists(csv_path):
            csv_path = self.download_dataset_from_hub()
            if csv_path is None:
                raise FileNotFoundError("Could not locate or download the dataset.")
        
        print(f"Loading dataset from: {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Use first 5000 rows for speed on Render free tier
        # (You can increase this to 10000 or 20000 if needed)
        df = df.head(5000)
        print(f"   Loaded {len(df)} rows")
        
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
        return self.user_ids[:limit] if self.user_ids else []

# Singleton instance
recommendation_engine = RecommendationEngine()
