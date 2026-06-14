from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import os
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
    # Check if dataset exists
    dataset_path = "Reviews.csv"  # Change this to your dataset filename
    
    if not os.path.exists(dataset_path):
        print(f"⚠️ Warning: Dataset '{dataset_path}' not found!")
        print("Please ensure your CSV file is in the backend folder")
        return
    
    try:
        recommendation_engine.load_and_train(dataset_path)
        print("✅ Recommendation engine ready!")
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
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)