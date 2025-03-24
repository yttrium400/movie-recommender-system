import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hybrid_recommender import HybridRecommender
from src.data_processing.data_loader import MovieDataLoader
from src.data_processing.preprocessor import MovieDataPreprocessor
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("test_hybrid_recommender", level="INFO")

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create sample user-movie matrix
    user_movie_data = {
        1: [5, 3, 0, 1],
        2: [4, 0, 0, 1],
        3: [1, 1, 0, 5],
        4: [1, 0, 0, 4],
        5: [0, 2, 4, 0],
    }
    user_movie_matrix = pd.DataFrame(
        user_movie_data,
        index=[1, 2, 3, 4],  # movie IDs
        columns=[1, 2, 3, 4, 5]  # user IDs
    ).T

    # Create sample movies data
    movies_data = pd.DataFrame({
        'movieId': [1, 2, 3, 4],
        'title': [
            'Movie 1',
            'Movie 2',
            'Movie 3',
            'Movie 4'
        ],
        'genres': [
            'Action|Adventure',
            'Comedy|Romance',
            'Drama|Thriller',
            'Action|Sci-Fi'
        ]
    })

    return user_movie_matrix, movies_data

@pytest.fixture
def real_data():
    """Load real MovieLens data for testing"""
    loader = MovieDataLoader()
    preprocessor = MovieDataPreprocessor()
    
    movies_df, ratings_df = loader.load_data()
    processed_movies = preprocessor.process_movie_metadata(movies_df)
    user_movie_matrix = preprocessor.create_user_movie_matrix(ratings_df)
    
    return user_movie_matrix, processed_movies

def test_hybrid_recommender_init():
    """Test hybrid recommender initialization"""
    # Test valid weights
    recommender = HybridRecommender(0.6, 0.4)
    assert recommender.collab_weight == 0.6
    assert recommender.content_weight == 0.4
    
    # Test invalid weights
    with pytest.raises(ValueError):
        HybridRecommender(0.6, 0.6)  # Sum > 1
    with pytest.raises(ValueError):
        HybridRecommender(-0.1, 1.1)  # Invalid range

def test_hybrid_recommender_fit(sample_data):
    """Test fitting the hybrid recommender"""
    user_movie_matrix, movies_data = sample_data
    recommender = HybridRecommender()
    
    # Test successful fit
    recommender.fit(user_movie_matrix, movies_data)
    
    # Test fit with invalid data
    with pytest.raises(ValueError):
        recommender.fit(pd.DataFrame(), movies_data)
    with pytest.raises(ValueError):
        recommender.fit(user_movie_matrix, pd.DataFrame())

def test_hybrid_recommendations(sample_data):
    """Test generating hybrid recommendations"""
    user_movie_matrix, movies_data = sample_data
    recommender = HybridRecommender()
    recommender.fit(user_movie_matrix, movies_data)
    
    # Test recommendations for a user
    user_id = 1
    user_ratings = {1: 5, 2: 3}
    recommendations = recommender.recommend_movies(user_id, user_ratings, n_recommendations=2)
    
    # Check recommendation format
    assert len(recommendations) == 2
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
    assert all(isinstance(movie_id, (int, np.integer)) and isinstance(score, float)
              for movie_id, score in recommendations)
    
    # Check scores are normalized and weighted
    assert all(0 <= score <= 1 for _, score in recommendations)
    
    # Check recommendations are sorted
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)

def test_weight_updates(sample_data):
    """Test weight update functionality"""
    user_movie_matrix, movies_data = sample_data
    recommender = HybridRecommender()
    recommender.fit(user_movie_matrix, movies_data)
    
    # Get initial recommendations
    user_id = 1
    user_ratings = {1: 5, 2: 3}
    recommendations = recommender.recommend_movies(user_id, user_ratings)
    
    # Test weight update with performance method
    initial_weights = (recommender.collab_weight, recommender.content_weight)
    recommender.update_weights(user_ratings, recommendations, method='performance')
    updated_weights = (recommender.collab_weight, recommender.content_weight)
    
    # Check weights still sum to 1
    assert abs(recommender.collab_weight + recommender.content_weight - 1.0) < 1e-6
    
    # Check weights have been updated
    assert initial_weights != updated_weights

def test_real_data_recommendations(real_data):
    """Test recommendations on real MovieLens data"""
    user_movie_matrix, movies_data = real_data
    recommender = HybridRecommender()
    recommender.fit(user_movie_matrix, movies_data)
    
    # Test recommendations for a user with some ratings
    user_id = user_movie_matrix.index[0]
    user_ratings = user_movie_matrix.iloc[0].to_dict()
    user_ratings = {k: v for k, v in user_ratings.items() if v > 0}
    
    recommendations = recommender.recommend_movies(user_id, user_ratings, n_recommendations=10)
    
    # Check recommendation quality
    assert len(recommendations) == 10
    assert len(set(movie_id for movie_id, _ in recommendations)) == 10  # No duplicates
    assert all(movie_id in movies_data['movieId'].values 
              for movie_id, _ in recommendations)  # Valid movie IDs
    
    logger.info("Hybrid recommender tests completed successfully") 