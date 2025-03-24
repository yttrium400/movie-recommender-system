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

@pytest.fixture
def sample_data():
    """Create sample user-movie rating matrix for testing"""
    data = {
        0: [5, 4, 1, 1, 0],
        1: [3, 0, 1, 0, 2],
        2: [0, 0, 0, 0, 4],
        3: [1, 1, 5, 4, 0],
        4: [4, 2, 0, 0, 1]
    }
    return pd.DataFrame(data, index=[1, 2, 3, 4, 5])

def test_collaborative_filtering_initialization():
    """Test initialization of CollaborativeFiltering class"""
    # Test with default k_neighbors
    cf = CollaborativeFiltering()
    assert cf.k_neighbors == 5
    
    # Test with custom k_neighbors
    cf = CollaborativeFiltering(k_neighbors=3)
    assert cf.k_neighbors == 3
    
    # Test with invalid k_neighbors
    with pytest.raises(ValueError):
        CollaborativeFiltering(k_neighbors=0)
    with pytest.raises(ValueError):
        CollaborativeFiltering(k_neighbors=-1)

def test_collaborative_filtering_fit(sample_data):
    """Test fitting of the collaborative filtering model"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Check if similarity matrices are created
    assert cf.user_similarity_matrix is not None
    assert cf.item_similarity_matrix is not None
    
    # Check dimensions
    assert cf.user_similarity_matrix.shape == (5, 5)
    assert cf.item_similarity_matrix.shape == (5, 5)
    
    # Test with empty matrix
    with pytest.raises(ValueError):
        cf.fit(pd.DataFrame())

def test_user_based_prediction(sample_data):
    """Test user-based prediction"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test prediction for known rating
    pred = cf.predict_user_based(1, 0)
    assert 0 <= pred <= 5
    
    # Test prediction for unknown rating
    pred = cf.predict_user_based(1, 2)
    assert 0 <= pred <= 5
    
    # Test with non-existent user
    with pytest.raises(KeyError):
        cf.predict_user_based(999, 0)
    
    # Test with non-existent movie
    with pytest.raises(KeyError):
        cf.predict_user_based(1, 999)

def test_item_based_prediction(sample_data):
    """Test item-based prediction"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test prediction for known rating
    pred = cf.predict_item_based(1, 0)
    assert 0 <= pred <= 5
    
    # Test prediction for unknown rating
    pred = cf.predict_item_based(1, 2)
    assert 0 <= pred <= 5
    
    # Test with non-existent user
    with pytest.raises(KeyError):
        cf.predict_item_based(999, 0)
    
    # Test with non-existent movie
    with pytest.raises(KeyError):
        cf.predict_item_based(1, 999)

def test_movie_recommendations(sample_data):
    """Test movie recommendations"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test user-based recommendations
    recs = cf.recommend_movies(1, n_recommendations=3, method='user')
    assert len(recs) <= 3
    for movie_id, rating in recs:
        assert 0 <= rating <= 5
    
    # Test item-based recommendations
    recs = cf.recommend_movies(1, n_recommendations=3, method='item')
    assert len(recs) <= 3
    for movie_id, rating in recs:
        assert 0 <= rating <= 5
    
    # Test with invalid method
    with pytest.raises(ValueError):
        cf.recommend_movies(1, method='invalid')
    
    # Test with non-existent user
    with pytest.raises(KeyError):
        cf.recommend_movies(999)

def test_cold_start_handling(sample_data):
    """Test handling of cold start problems"""
    cf = CollaborativeFiltering(k_neighbors=2)
    
    # Add a new user with no ratings
    new_user_data = pd.Series([0, 0, 0, 0, 0], name=6)
    sample_data = pd.concat([sample_data, pd.DataFrame([new_user_data])])
    
    cf.fit(sample_data)
    
    # Test predictions for new user
    pred = cf.predict_user_based(6, 0)
    assert pred == 0.0
    
    pred = cf.predict_item_based(6, 0)
    assert pred == 0.0
    
    # Test recommendations for new user
    recs = cf.recommend_movies(6, n_recommendations=3)
    assert len(recs) == 3
    for _, rating in recs:
        assert rating == 0.0

def test_real_data():
    """Test with real MovieLens data"""
    # Create a small subset of MovieLens-style data
    users = range(1, 11)
    movies = range(1, 21)
    np.random.seed(42)
    
    data = {}
    for movie in movies:
        ratings = np.zeros(len(users))
        # Randomly assign ratings (1-5) to 60% of user-movie pairs
        mask = np.random.choice([0, 1], size=len(users), p=[0.4, 0.6])
        ratings[mask == 1] = np.random.randint(1, 6, size=sum(mask))
        data[movie] = ratings
    
    df = pd.DataFrame(data, index=users)
    
    # Test collaborative filtering with this data
    cf = CollaborativeFiltering(k_neighbors=3)
    cf.fit(df)
    
    # Test recommendations for a few users
    for user_id in [1, 5, 10]:
        recs = cf.recommend_movies(user_id, n_recommendations=5)
        assert len(recs) <= 5
        for movie_id, rating in recs:
            assert 0 <= rating <= 5

if __name__ == "__main__":
    logger.info("Starting Sprint 2 tests...")
    
    try:
        test_collaborative_filtering_initialization()
        test_collaborative_filtering_fit()
        test_user_based_prediction()
        test_item_based_prediction()
        test_movie_recommendations()
        test_cold_start_handling()
        test_real_data()
        logger.info("All Sprint 2 tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 