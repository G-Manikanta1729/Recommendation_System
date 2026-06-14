from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import sys
from model import recommendation_engine

# Initialize FastAPI app
app = FastAPI(title="Amazon Style Recommendation System", 
              description="Product recommendation engine using collaborative filtering",
              version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5

class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    confidence: float

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load and train the recommendation model when server starts"""
    print("🔄 Loading recommendation model...")
    
    # The model's load_and_train method now handles:
    # 1. Checking for local dataset
    # 2. Downloading from Hugging Face if not found
    # 3. Training the model
    
    try:
        # Call without arguments - the model will handle dataset location
        success = recommendation_engine.load_and_train()
        
        if success:
            print("✅ Recommendation engine ready!")
            if recommendation_engine.user_ids:
                print(f"   - Users loaded: {len(recommendation_engine.user_ids)}")
                print(f"   - Products: {len(recommendation_engine.user_product_matrix.columns) if recommendation_engine.user_product_matrix is not None else 0}")
        else:
            print("⚠️ Model could not be loaded")
            
    except FileNotFoundError as e:
        print(f"❌ Dataset error: {str(e)}")
        print("   Please ensure the dataset is available or check Hugging Face connection")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Amazon Style Recommendation System API",
        "status": "active",
        "endpoints": {
            "/recommend/{user_id}": "Get recommendations for a user",
            "/users": "Get list of available users",
            "/health": "Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the API is working"""
    return {
        "status": "healthy",
        "model_loaded": recommendation_engine.user_product_matrix is not None,
        "total_users": len(recommendation_engine.user_ids) if recommendation_engine.user_ids else 0
    }

@app.get("/users")
async def get_users(limit: int = 20):
    """Get list of available users"""
    if not recommendation_engine.user_ids:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    users = recommendation_engine.user_ids[:limit]
    return {
        "total_users": len(recommendation_engine.user_ids),
        "users": users
    }

@app.get("/recommend/{user_id}")
async def recommend_products(user_id: str, limit: int = 5):
    """Get product recommendations for a specific user"""
    if not recommendation_engine.user_ids:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait for dataset to download.")
    
    recommendations = recommendation_engine.get_recommendations(user_id, limit)
    
    if isinstance(recommendations, dict) and "error" in recommendations:
        raise HTTPException(status_code=404, detail=recommendations["error"])
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    }

@app.post("/recommend")
async def recommend_products_post(request: RecommendationRequest):
    """Get recommendations using POST method"""
    return await recommend_products(request.user_id, request.num_recommendations)

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
