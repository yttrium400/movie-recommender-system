import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.collaborative_filtering import CollaborativeFiltering
from src.data_processing.data_loader import MovieDataLoader
from src.data_processing.preprocessor import MovieDataPreprocessor
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

@pytest.fixture
def real_data():
    """Load real MovieLens data for testing"""
    loader = MovieDataLoader()
    preprocessor = MovieDataPreprocessor()
    
    movies_df, ratings_df = loader.load_data()
    processed_ratings = preprocessor.process_ratings(ratings_df)
    train, val, _ = preprocessor.split_data(processed_ratings, method='time')
    
    user_movie_matrix = preprocessor.create_user_movie_matrix(train)
    return user_movie_matrix, val

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
    
    # Check similarity matrix properties
    assert np.allclose(cf.user_similarity_matrix.diagonal(), 1.0)
    assert np.allclose(cf.user_similarity_matrix, cf.user_similarity_matrix.T)
    assert np.allclose(cf.item_similarity_matrix, cf.item_similarity_matrix.T)
    
    # Test with empty data
    with pytest.raises(ValueError):
        cf.fit(pd.DataFrame())

def test_user_based_prediction(sample_data):
    """Test user-based prediction"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test prediction for known rating
    pred = cf.predict_user_based(1, 0)
    assert isinstance(pred, float)
    assert 0 <= pred <= 5
    
    # Test prediction for unknown rating
    pred = cf.predict_user_based(1, 2)
    assert isinstance(pred, float)
    assert 0 <= pred <= 5
    
    # Test with non-existent user
    with pytest.raises(KeyError):
        cf.predict_user_based(999, 0)
    
    # Test with non-existent movie
    with pytest.raises(KeyError):
        cf.predict_user_based(1, 999)
    
    # Test cold start handling
    new_user_data = pd.Series([0, 0, 0, 0, 0], name=6)
    new_data = pd.concat([sample_data, pd.DataFrame([new_user_data])])
    cf.fit(new_data)
    assert cf.predict_user_based(6, 0) == 0.0

def test_item_based_prediction(sample_data):
    """Test item-based prediction"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test prediction for known rating
    pred = cf.predict_item_based(1, 0)
    assert isinstance(pred, float)
    assert 0 <= pred <= 5
    
    # Test prediction for unknown rating
    pred = cf.predict_item_based(1, 2)
    assert isinstance(pred, float)
    assert 0 <= pred <= 5
    
    # Test with non-existent user
    with pytest.raises(KeyError):
        cf.predict_item_based(999, 0)
    
    # Test with non-existent movie
    with pytest.raises(KeyError):
        cf.predict_item_based(1, 999)
    
    # Test cold start handling
    new_user_data = pd.Series([0, 0, 0, 0, 0], name=6)
    new_data = pd.concat([sample_data, pd.DataFrame([new_user_data])])
    cf.fit(new_data)
    assert cf.predict_item_based(6, 0) == 0.0

def test_movie_recommendations(sample_data):
    """Test movie recommendations"""
    cf = CollaborativeFiltering(k_neighbors=2)
    cf.fit(sample_data)
    
    # Test user-based recommendations
    recs = cf.recommend_movies(1, n_recommendations=3, method='user')
    assert len(recs) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    assert all(0 <= rating <= 5 for _, rating in recs)
    
    # Test item-based recommendations
    recs = cf.recommend_movies(1, n_recommendations=3, method='item')
    assert len(recs) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    assert all(0 <= rating <= 5 for _, rating in recs)
    
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

def test_real_data(real_data):
    """Test with real MovieLens data"""
    user_movie_matrix, validation_data = real_data
    cf = CollaborativeFiltering(k_neighbors=10)
    cf.fit(user_movie_matrix)
    
    # Test predictions on validation set
    predictions = []
    actuals = []
    
    # Sample a subset of validation data for testing
    sample_size = min(1000, len(validation_data))
    validation_sample = validation_data.sample(n=sample_size, random_state=42)
    
    for _, row in validation_sample.iterrows():
        try:
            pred = cf.predict_user_based(row['userId'], row['movieId'])
            if pred > 0:  # Only consider non-zero predictions
                predictions.append(pred)
                actuals.append(row['rating'])
        except (ValueError, KeyError):
            continue
    
    if predictions:
        rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)]))
        assert rmse <= 1.5, "RMSE too high on validation data"
    
    logger.info("Real data test passed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 