import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import MovieDataLoader
from src.models.collaborative_filtering import CollaborativeFiltering
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("test_sprint2", level="INFO")

def test_collaborative_filtering_initialization():
    """Test collaborative filtering model initialization"""
    cf = CollaborativeFiltering(k_neighbors=5)
    assert cf.k_neighbors == 5, "Wrong number of neighbors"
    assert cf.user_similarity_matrix is None, "User similarity matrix should be None before fitting"
    assert cf.item_similarity_matrix is None, "Item similarity matrix should be None before fitting"
    
    logger.info("Collaborative filtering initialization test passed successfully")

def test_collaborative_filtering_fit():
    """Test fitting the collaborative filtering model"""
    # Create sample user-movie matrix
    data = {
        1: [5, 3, 0, 1],
        2: [4, 0, 0, 1],
        3: [1, 1, 0, 5],
        4: [1, 0, 0, 4]
    }
    user_movie_matrix = pd.DataFrame(data).T
    
    # Fit model
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(user_movie_matrix)
    
    # Check similarity matrices
    assert cf.user_similarity_matrix is not None, "User similarity matrix not computed"
    assert cf.item_similarity_matrix is not None, "Item similarity matrix not computed"
    assert cf.user_similarity_matrix.shape == (4, 4), "Wrong user similarity matrix shape"
    assert cf.item_similarity_matrix.shape == (4, 4), "Wrong item similarity matrix shape"
    
    logger.info("Collaborative filtering fit test passed successfully")

def test_user_based_prediction():
    """Test user-based collaborative filtering predictions"""
    # Create sample user-movie matrix
    data = {
        1: [5, 3, 0, 1],
        2: [4, 0, 0, 1],
        3: [1, 1, 0, 5],
        4: [1, 0, 0, 4]
    }
    user_movie_matrix = pd.DataFrame(data).T
    
    # Fit model and make prediction
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(user_movie_matrix)
    
    # Test prediction
    prediction = cf.predict_user_based(1, 2)  # Predict rating for user 1, movie 2
    assert isinstance(prediction, float), "Prediction should be a float"
    assert 0 <= prediction <= 5, "Prediction should be between 0 and 5"
    
    logger.info("User-based prediction test passed successfully")

def test_item_based_prediction():
    """Test item-based collaborative filtering predictions"""
    # Create sample user-movie matrix
    data = {
        1: [5, 3, 0, 1],
        2: [4, 0, 0, 1],
        3: [1, 1, 0, 5],
        4: [1, 0, 0, 4]
    }
    user_movie_matrix = pd.DataFrame(data).T
    
    # Fit model and make prediction
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(user_movie_matrix)
    
    # Test prediction
    prediction = cf.predict_item_based(1, 2)  # Predict rating for user 1, movie 2
    assert isinstance(prediction, float), "Prediction should be a float"
    assert 0 <= prediction <= 5, "Prediction should be between 0 and 5"
    
    logger.info("Item-based prediction test passed successfully")

def test_movie_recommendations():
    """Test movie recommendation functionality"""
    # Create sample user-movie matrix
    data = {
        1: [5, 3, 0, 1],
        2: [4, 0, 0, 1],
        3: [1, 1, 0, 5],
        4: [1, 0, 0, 4]
    }
    user_movie_matrix = pd.DataFrame(data).T
    
    # Fit model and get recommendations
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(user_movie_matrix)
    
    # Test user-based recommendations
    user_recs = cf.recommend_movies(1, n_recommendations=2, method='user')
    assert len(user_recs) <= 2, "Wrong number of user-based recommendations"
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in user_recs), "Invalid recommendation format"
    
    # Test item-based recommendations
    item_recs = cf.recommend_movies(1, n_recommendations=2, method='item')
    assert len(item_recs) <= 2, "Wrong number of item-based recommendations"
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in item_recs), "Invalid recommendation format"
    
    logger.info("Movie recommendations test passed successfully")

def test_real_data():
    """Test collaborative filtering with real MovieLens data"""
    # Load real data
    loader = MovieDataLoader()
    user_movie_matrix = loader.get_user_movie_matrix()
    
    # Fit model
    cf = CollaborativeFiltering(k_neighbors=10)
    cf.fit(user_movie_matrix)
    
    # Get recommendations for a random user
    user_id = user_movie_matrix.index[0]
    recommendations = cf.recommend_movies(user_id, n_recommendations=5)
    
    assert len(recommendations) <= 5, "Wrong number of recommendations"
    assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations), "Invalid recommendation format"
    
    logger.info("Real data test passed successfully")

if __name__ == "__main__":
    logger.info("Starting Sprint 2 tests...")
    
    try:
        test_collaborative_filtering_initialization()
        test_collaborative_filtering_fit()
        test_user_based_prediction()
        test_item_based_prediction()
        test_movie_recommendations()
        test_real_data()
        logger.info("All Sprint 2 tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 