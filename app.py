from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from feature_processor import FeatureProcessor
from recommendation_engine import RecommendationEngine

app = FastAPI(
    title="ML Recommendation Service",
    description="Secure API for user recommendations",
    version="1.0.0"
)

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["https://www.dyce.in"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Missing API KEY")


feature_processing = FeatureProcessor()
recommendation_engine = RecommendationEngine(feature_processing)

class UserData(BaseModel):
    user_id: str = Field(..., description="User ID")
    gender: str = Field(..., description="User gender")
    interests: List[str] = Field(..., description="List of user interests")
    campusVibeTags: List[str] = Field(..., description="Campus vibe tags")
    hangoutSpot: str = Field(..., description="Preferred hangout spot")
    preferences: str = Field(..., description="User preferences")
    prompts_1: str = Field(..., description="First prompt response")
    prompts_2: str = Field(..., description="Second prompt response")
    prompts_3: str = Field(..., description="Third prompt response")
    
    name: Optional[str] = None
    bio: Optional[str] = None
    age: Optional[int] = Field(None, ge=18, le=120)
    location: Optional[str] = None
    
    class Config:
        extra = "allow"

class UpdateUserVectorRequest(BaseModel):
    user_data: UserData = Field(..., description="User data for vector creation")

class GetRecommendationsRequest(BaseModel):
    target_user_id: str = Field(..., description="ID of the target user")
    all_users_vector_data: List[UserData] = Field(..., description="List of all users vector data")
    recommendation_history: Dict[str, int] = Field(..., description="Dictionary of all recommended users")
    liked_users: List[int] = Field(..., description="List of all liked users")
    n_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")

class UserVectorResponse(BaseModel):
    user_id: str
    gender: str
    vector: List[int]

class RecommendationsResponse(BaseModel):
    target_user_id: str
    recommendations: List[str]
    similarity_scores: List[int]
    count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
  
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.post("/convert_to_user_vector", response_model=UserVectorResponse)
# INPUT: single user_data that is a dictionary
async def convert_to_user_vector(
    request: UpdateUserVectorRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        user_data = request.user_data

        user_vector = feature_processing.create_user_vector(user_data)
        
        return UserVectorResponse(
            user_id=user_data['id'],
            gender=user_data['gender'],
            vector=user_vector
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    
@app.post("/get_recommendations", response_model=RecommendationsResponse)
# INPUT: target_user_id i.e a string
# all_user_vector_data that is a list of dictionary that has user_id, gender, vector as keys
# recommendation_history that is a dictionary having user_id as key, and number of time recommended as value
# liked_users that is a list of user ids
# n_recommendations that is number of recommendations to fetch
async def get_recommendations(
    request: GetRecommendationsRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        all_users_vector_data = [user for user in request.all_users_vector_data]

        recommendations = recommendation_engine.get_recommendations(
            request.target_user_id,
            all_users_vector_data,
            request.recommendation_history,
            request.liked_users,
            request.n_recommendations
        )
        
        return RecommendationsResponse(
            target_user_id=request.target_user_id,
            recommended_user_ids=list(recommendations.keys()),
            similarity_scores=list(recommendations.values()),
            count=len(recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": "Internal server error",
        "status_code": 500
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )