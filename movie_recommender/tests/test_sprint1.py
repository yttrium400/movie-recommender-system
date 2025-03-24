import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import MovieDataLoader
from src.data_processing.preprocessor import MovieDataPreprocessor
from src.utils.config import Config
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("test_sprint1", level="INFO")

@pytest.fixture
def data_loader():
    """Create a data loader instance"""
    return MovieDataLoader()

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    return MovieDataPreprocessor()

def test_data_loading(data_loader):
    """Test data loading functionality"""
    movies_df, ratings_df = data_loader.load_data()
    
    # Check if dataframes are not empty
    assert not movies_df.empty, "Movies dataframe is empty"
    assert not ratings_df.empty, "Ratings dataframe is empty"
    
    # Check required columns
    assert all(col in movies_df.columns for col in ['movieId', 'title', 'genres'])
    assert all(col in ratings_df.columns for col in ['userId', 'movieId', 'rating', 'timestamp'])
    
    # Verify data types
    assert movies_df['movieId'].dtype in ['int32', 'int64']
    assert isinstance(movies_df['title'].iloc[0], str)
    assert isinstance(movies_df['genres'].iloc[0], str)
    
    assert ratings_df['userId'].dtype in ['int32', 'int64']
    assert ratings_df['movieId'].dtype in ['int32', 'int64']
    assert ratings_df['rating'].dtype in ['float32', 'float64']
    assert ratings_df['timestamp'].dtype in ['int32', 'int64']
    
    # Check value ranges
    assert ratings_df['rating'].between(0, 5).all(), "Ratings should be between 0 and 5"
    assert movies_df['movieId'].is_unique, "Movie IDs should be unique"
    
    logger.info("Data loading test passed successfully")

def test_data_preprocessing(data_loader, preprocessor):
    """Test data preprocessing functionality"""
    # Load data
    movies_df, ratings_df = data_loader.load_data()
    
    # Test movie metadata processing
    processed_movies = preprocessor.process_movie_metadata(movies_df)
    
    # Check year extraction
    assert 'year' in processed_movies.columns, "Year column not created"
    assert processed_movies['year'].dtype in ['int32', 'int64']
    assert processed_movies['year'].between(1900, datetime.now().year).all()
    
    # Check genre processing
    assert processed_movies['genres'].notna().all(), "Missing values in genres"
    assert all(isinstance(genres, str) for genres in processed_movies['genres'])
    
    # Check genre indicator columns
    common_genres = {'Action', 'Adventure', 'Comedy', 'Drama', 'Romance'}
    for genre in common_genres:
        assert genre in processed_movies.columns, f"Missing genre indicator column: {genre}"
        assert processed_movies[genre].dtype == bool, f"Genre column {genre} should be boolean"
    
    # Test ratings processing
    processed_ratings = preprocessor.process_ratings(ratings_df)
    
    # Check rating normalization
    assert 'rating_normalized' in processed_ratings.columns
    assert processed_ratings['rating_normalized'].between(0, 1).all()
    
    # Check timestamp processing
    assert all(col in processed_ratings.columns for col in ['day_of_week', 'hour_of_day'])
    assert processed_ratings['day_of_week'].between(0, 6).all()
    assert processed_ratings['hour_of_day'].between(0, 23).all()
    
    logger.info("Data preprocessing test passed successfully")

def test_data_splitting(data_loader, preprocessor):
    """Test data splitting functionality"""
    # Load and preprocess data
    movies_df, ratings_df = data_loader.load_data()
    processed_ratings = preprocessor.process_ratings(ratings_df)
    
    # Test chronological split
    train, val, test = preprocessor.split_data(processed_ratings, method='time')
    
    # Check split sizes
    total_size = len(processed_ratings)
    assert len(train) + len(val) + len(test) == total_size
    assert 0.6 <= len(train) / total_size <= 0.8
    assert 0.1 <= len(val) / total_size <= 0.2
    assert 0.1 <= len(test) / total_size <= 0.2
    
    # Check chronological ordering
    assert train['timestamp'].max() <= val['timestamp'].min()
    assert val['timestamp'].max() <= test['timestamp'].min()
    
    # Test random split
    train2, val2, test2 = preprocessor.split_data(processed_ratings, method='random')
    
    # Check split sizes for random split
    assert len(train2) + len(val2) + len(test2) == total_size
    assert 0.6 <= len(train2) / total_size <= 0.8
    assert 0.1 <= len(val2) / total_size <= 0.2
    assert 0.1 <= len(test2) / total_size <= 0.2
    
    logger.info("Data splitting test passed successfully")

def test_user_movie_matrix(data_loader, preprocessor):
    """Test user-movie matrix creation"""
    movies_df, ratings_df = data_loader.load_data()
    processed_ratings = preprocessor.process_ratings(ratings_df)
    
    # Create user-movie matrix
    user_movie_matrix = preprocessor.create_user_movie_matrix(processed_ratings)
    
    # Check matrix properties
    assert isinstance(user_movie_matrix, pd.DataFrame)
    assert user_movie_matrix.index.name == 'userId'
    assert all(movie_id in movies_df['movieId'].values for movie_id in user_movie_matrix.columns)
    
    # Check value ranges
    assert user_movie_matrix.min().min() >= 0
    assert user_movie_matrix.max().max() <= 5
    
    # Check sparsity
    sparsity = 1.0 - (user_movie_matrix.notna().sum().sum() / 
                      (user_movie_matrix.shape[0] * user_movie_matrix.shape[1]))
    assert 0.90 <= sparsity <= 0.999, "Unexpected sparsity level"
    
    logger.info("User-movie matrix creation test passed successfully")

def test_config():
    """Test configuration management"""
    config = Config()
    
    # Check required configuration sections
    assert hasattr(config, 'data_paths')
    assert hasattr(config, 'preprocessing')
    assert hasattr(config, 'logging')
    
    # Check data paths
    assert 'movies_file' in config.data_paths
    assert 'ratings_file' in config.data_paths
    
    # Check preprocessing settings
    assert 'train_size' in config.preprocessing
    assert 'validation_size' in config.preprocessing
    assert 'test_size' in config.preprocessing
    assert 'random_state' in config.preprocessing
    
    logger.info("Configuration test passed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 