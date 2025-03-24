import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import MovieDataLoader
from src.data_processing.preprocessor import MovieDataPreprocessor
from src.utils.config import Config
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("test_sprint1", level="INFO")

def test_data_loading():
    """Test data loading functionality"""
    loader = MovieDataLoader()
    movies_df, ratings_df = loader.load_data()
    
    # Check if dataframes are not empty
    assert not movies_df.empty, "Movies dataframe is empty"
    assert not ratings_df.empty, "Ratings dataframe is empty"
    
    # Check required columns
    assert all(col in movies_df.columns for col in ['movieId', 'title', 'genres'])
    assert all(col in ratings_df.columns for col in ['userId', 'movieId', 'rating', 'timestamp'])
    
    logger.info("Data loading test passed successfully")

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    loader = MovieDataLoader()
    preprocessor = MovieDataPreprocessor()
    
    # Load data
    movies_df, ratings_df = loader.load_data()
    
    # Test movie metadata processing
    processed_movies = preprocessor.process_movie_metadata(movies_df)
    assert 'year' in processed_movies.columns, "Year column not created"
    assert processed_movies.iloc[:, 2].str.strip().str.len().gt(0).all(), "Empty values in genres"
    
    # Test ratings processing
    processed_ratings = preprocessor.process_ratings(ratings_df)
    assert 'rating_normalized' in processed_ratings.columns, "Normalized ratings not created"
    assert 'day_of_week' in processed_ratings.columns, "Day of week not created"
    assert 'hour_of_day' in processed_ratings.columns, "Hour of day not created"
    
    # Test content features
    content_features = preprocessor.create_content_features(movies_df)
    assert content_features.shape[0] == len(movies_df), "Wrong number of content features"
    
    logger.info("Data preprocessing test passed successfully")

def test_user_movie_matrix():
    """Test user-movie matrix creation"""
    loader = MovieDataLoader()
    
    # Get user-movie matrix
    user_movie_matrix = loader.get_user_movie_matrix()
    
    # Check matrix properties
    assert isinstance(user_movie_matrix, pd.DataFrame), "Result is not a DataFrame"
    assert user_movie_matrix.shape[0] > 0, "Matrix has no users"
    assert user_movie_matrix.shape[1] > 0, "Matrix has no movies"
    assert not user_movie_matrix.isna().any().any(), "Matrix contains missing values"
    
    logger.info("User-movie matrix test passed successfully")

def test_config():
    """Test configuration functionality"""
    config = Config()
    
    # Test paths
    assert config.get_data_path().exists(), "Data path not created"
    assert config.get_models_path().exists(), "Models path not created"
    assert config.get_logs_path().exists(), "Logs path not created"
    
    # Test model parameters
    collab_params = config.get_model_params("collaborative")
    assert collab_params is not None, "Collaborative parameters not found"
    assert "n_factors" in collab_params, "Missing n_factors parameter"
    
    # Test config save and load
    config.save_config("test_config.yaml")
    loaded_config = Config.load_config("test_config.yaml")
    assert loaded_config.model_params == config.model_params, "Config save/load mismatch"
    
    logger.info("Configuration test passed successfully")

if __name__ == "__main__":
    logger.info("Starting Sprint 1 tests...")
    
    try:
        test_data_loading()
        test_data_preprocessing()
        test_user_movie_matrix()
        test_config()
        logger.info("All Sprint 1 tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 