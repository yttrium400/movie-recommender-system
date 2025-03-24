from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import jwt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.data_loader import MovieDataLoader
from data_processing.preprocessor import MovieDataPreprocessor
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based.content_based_filtering import ContentBasedFiltering

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="API for movie recommendations using collaborative and content-based filtering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
SECRET_KEY = "your-secret-key"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize models and data
data_loader = MovieDataLoader()
preprocessor = MovieDataPreprocessor()
movies_df, ratings_df = data_loader.load_data()
processed_movies = preprocessor.process_movie_metadata(movies_df)

# Initialize recommender models
collaborative_model = CollaborativeFiltering()
content_based_model = ContentBasedFiltering()

# Fit content-based model
content_based_model.fit(processed_movies)

# Create user-movie matrix for collaborative filtering
user_movie_matrix = preprocessor.create_user_movie_matrix(ratings_df)
collaborative_model.fit(user_movie_matrix)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None

class MovieRating(BaseModel):
    movie_id: int
    rating: float

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float

# Helper functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return User(username=username)
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# API endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, validate against a user database
    user = User(username=form_data.username)
    access_token = create_access_token({"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")

@app.get("/movies/", response_model=List[dict])
async def get_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """Get a paginated list of movies."""
    movies = processed_movies.iloc[skip:skip + limit]
    return movies.to_dict('records')

@app.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details of a specific movie."""
    movie = processed_movies[processed_movies['movieId'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie.iloc[0].to_dict()

@app.get("/movies/{movie_id}/similar")
async def get_similar_movies(
    movie_id: int,
    n: int = Query(5, ge=1, le=20)
):
    """Get similar movies based on content."""
    try:
        similar_movies = content_based_model.get_similar_movies(movie_id, n=n)
        result = []
        for movie_id, score in similar_movies:
            movie = processed_movies[processed_movies['movieId'] == movie_id].iloc[0]
            result.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'score': score
            })
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/users/me/ratings")
async def rate_movie(
    rating: MovieRating,
    current_user: User = Depends(get_current_user)
):
    """Submit a movie rating."""
    # In production, save to a database
    return {"status": "success", "message": "Rating submitted"}

@app.get("/users/me/recommendations")
async def get_recommendations(
    current_user: User = Depends(get_current_user),
    method: str = Query("collaborative", regex="^(collaborative|content|hybrid)$"),
    n: int = Query(5, ge=1, le=20)
):
    """Get personalized movie recommendations."""
    # In production, get user's actual ratings from database
    # For demo, use some sample ratings
    sample_ratings = {1: 5.0, 2: 4.0, 3: 3.0}
    
    if method == "collaborative":
        recommendations = collaborative_model.recommend_movies(1, n_recommendations=n)  # Using user_id=1 for demo
    elif method == "content":
        recommendations = content_based_model.recommend_movies(sample_ratings, n_recommendations=n)
    else:  # hybrid
        collab_recs = collaborative_model.recommend_movies(1, n_recommendations=n)
        content_recs = content_based_model.recommend_movies(sample_ratings, n_recommendations=n)
        # Combine and sort by score
        all_recs = collab_recs + content_recs
        all_recs.sort(key=lambda x: x[1], reverse=True)
        recommendations = all_recs[:n]
    
    result = []
    for movie_id, score in recommendations:
        movie = processed_movies[processed_movies['movieId'] == movie_id].iloc[0]
        result.append({
            'movie_id': movie_id,
            'title': movie['title'],
            'genres': movie['genres'],
            'score': score
        })
    return result

@app.get("/search")
async def search_movies(
    query: str,
    limit: int = Query(10, ge=1, le=100)
):
    """Search for movies by title or genre."""
    # Simple case-insensitive search
    mask = (
        processed_movies['title'].str.contains(query, case=False) |
        processed_movies['genres'].str.contains(query, case=False)
    )
    results = processed_movies[mask].head(limit)
    return results.to_dict('records')

@app.get("/users/me/ratings")
async def get_user_ratings(current_user: User = Depends(get_current_user)):
    """Get all movies rated by the current user."""
    try:
        # In production, get user's actual ratings from database
        # For demo, use some sample ratings
        sample_ratings = {1: 5.0, 2: 4.0, 3: 3.0}
        
        result = []
        for movie_id, rating in sample_ratings.items():
            movie = processed_movies[processed_movies['movieId'] == movie_id].iloc[0]
            result.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'year': movie.get('year'),
                'rating': rating
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 