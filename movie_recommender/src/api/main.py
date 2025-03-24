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

# In-memory storage for user data (for demo purposes)
user_ratings = {}  # Dictionary to store user ratings: {username: {movie_id: rating}}
user_profiles = {}  # Dictionary to store user profiles: {username: UserProfile}

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
    full_name: Optional[str] = None
    preferred_genres: Optional[List[str]] = None
    profile_complete: bool = False

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserProfile(BaseModel):
    full_name: Optional[str] = None
    preferred_genres: List[str]

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
    username = current_user.username
    
    # Initialize user ratings dict if it doesn't exist
    if username not in user_ratings:
        user_ratings[username] = {}
    
    # Store the rating
    user_ratings[username][rating.movie_id] = rating.rating
    
    return {"status": "success", "message": "Rating submitted"}

@app.get("/users/me/ratings", response_model=List[dict])
async def get_user_ratings(current_user: User = Depends(get_current_user)):
    """Get all ratings submitted by the current user."""
    username = current_user.username
    
    if username not in user_ratings or not user_ratings[username]:
        return []
    
    result = []
    for movie_id, rating_value in user_ratings[username].items():
        try:
            movie = processed_movies[processed_movies['movieId'] == movie_id].iloc[0]
            result.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'rating': rating_value
            })
        except IndexError:
            # Skip if movie not found
            continue
    
    return result

@app.get("/users/me/recommendations")
async def get_recommendations(
    current_user: User = Depends(get_current_user),
    method: str = Query("collaborative", regex="^(collaborative|content|hybrid)$"),
    n: int = Query(5, ge=1, le=20)
):
    """Get personalized movie recommendations."""
    username = current_user.username
    
    # Get user's actual ratings or use sample ratings
    user_movie_ratings = {}
    if username in user_ratings:
        user_movie_ratings = user_ratings[username]
    
    # If no ratings yet, use sample data
    if not user_movie_ratings:
        user_movie_ratings = {1: 5.0, 2: 4.0, 3: 3.0}
    
    if method == "collaborative":
        # For demo, just use user_id=1, in production would map username to user_id
        recommendations = collaborative_model.recommend_movies(1, n_recommendations=n)
    elif method == "content":
        recommendations = content_based_model.recommend_movies(user_movie_ratings, n_recommendations=n)
    else:  # hybrid
        collab_recs = collaborative_model.recommend_movies(1, n_recommendations=n)
        content_recs = content_based_model.recommend_movies(user_movie_ratings, n_recommendations=n)
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

@app.post("/users/register")
async def register_user(user: UserCreate):
    """Register a new user."""
    # In production, check if user exists and hash the password
    # For demo, just return a success message
    access_token = create_access_token({"sub": user.username})
    return {"status": "success", "message": "User registered successfully", "access_token": access_token, "token_type": "bearer"}

@app.post("/users/me/profile")
async def create_user_profile(
    profile: UserProfile,
    current_user: User = Depends(get_current_user)
):
    """Create or update user profile."""
    # Validate that exactly 5 genres are selected
    if len(profile.preferred_genres) != 5:
        raise HTTPException(
            status_code=400, 
            detail="Please select exactly 5 favorite genres"
        )
    
    username = current_user.username
    
    # Store the profile
    user_profiles[username] = profile
    
    # For demo, return user model with updated profile
    updated_user = User(
        username=current_user.username,
        email=current_user.email,
        full_name=profile.full_name,
        preferred_genres=profile.preferred_genres,
        profile_complete=True
    )
    
    return updated_user

@app.get("/users/me")
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile information."""
    username = current_user.username
    
    # Get stored profile if it exists
    profile = user_profiles.get(username)
    
    if profile:
        # Return user with profile data
        return User(
            username=current_user.username,
            email=current_user.email,
            full_name=profile.full_name,
            preferred_genres=profile.preferred_genres,
            profile_complete=True
        )
    
    # Otherwise return basic user info
    return current_user

@app.get("/genres")
async def get_available_genres():
    """Get a list of all available movie genres."""
    # Extract unique genres from the dataset
    all_genres = []
    for genres_str in processed_movies['genres'].unique():
        genres = genres_str.split('|')
        all_genres.extend(genres)
    
    # Get unique genres and sort
    unique_genres = sorted(list(set(all_genres)))
    return unique_genres

@app.get("/users/me/genre-recommendations")
async def get_genre_based_recommendations(
    current_user: User = Depends(get_current_user),
    n: int = Query(10, ge=1, le=20)
):
    """Get movie recommendations based on user's preferred genres."""
    if not current_user.preferred_genres:
        raise HTTPException(
            status_code=400,
            detail="Please complete your profile with preferred genres first"
        )
    
    # Filter movies that match user's preferred genres
    genre_matches = []
    for _, movie in processed_movies.iterrows():
        movie_genres = set(movie['genres'].split('|'))
        user_genres = set(current_user.preferred_genres)
        # Calculate how many of the user's preferred genres match this movie
        match_count = len(movie_genres.intersection(user_genres))
        if match_count > 0:
            genre_matches.append((movie, match_count))
    
    # Sort by number of matching genres (descending)
    genre_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N recommendations
    recommendations = genre_matches[:n]
    
    result = []
    for movie, match_count in recommendations:
        result.append({
            'movie_id': movie['movieId'],
            'title': movie['title'],
            'genres': movie['genres'],
            'score': match_count  # Score is the number of matching genres
        })
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 