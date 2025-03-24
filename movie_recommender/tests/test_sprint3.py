import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.content_based.content_based_filtering import ContentBasedFiltering
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("test_sprint3", level="INFO")

@pytest.fixture
def sample_movies():
    """Create sample movie data for testing"""
    data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': [
            'The Shawshank Redemption',
            'The Godfather',
            'The Dark Knight',
            'Inception',
            'Pulp Fiction'
        ],
        'genres': [
            'Drama|Crime',
            'Drama|Crime',
            'Action|Crime|Drama',
            'Action|Sci-Fi|Thriller',
            'Crime|Drama|Thriller'
        ],
        'tags': [
            'prison escape redemption',
            'mafia family crime',
            'superhero dark vigilante',
            'dreams mind-bending sci-fi',
            'nonlinear crime dialogue'
        ]
    }
    return pd.DataFrame(data)

def test_content_based_initialization():
    """Test initialization of ContentBasedFiltering class"""
    cbf = ContentBasedFiltering()
    assert cbf.genre_features is None
    assert cbf.tag_features is None
    assert cbf.movie_ids is None
    assert cbf.genre_encoder is not None
    assert cbf.tag_vectorizer is not None

def test_feature_processing(sample_movies):
    """Test feature processing methods"""
    cbf = ContentBasedFiltering()
    
    # Test genre processing
    genre_features = cbf._process_genres(sample_movies['genres'])
    assert isinstance(genre_features, np.ndarray)
    assert genre_features.shape[0] == len(sample_movies)
    
    # Test tag processing
    tag_features = cbf._process_tags(sample_movies['tags'])
    assert isinstance(tag_features, np.ndarray)
    assert tag_features.shape[0] == len(sample_movies)

def test_model_fitting(sample_movies):
    """Test model fitting"""
    cbf = ContentBasedFiltering()
    cbf.fit(sample_movies)
    
    # Check if features are created
    assert cbf.genre_features is not None
    assert cbf.tag_features is not None
    assert cbf.movie_ids is not None
    
    # Check dimensions
    assert len(cbf.movie_ids) == len(sample_movies)
    assert cbf.genre_features.shape[0] == len(sample_movies)
    assert cbf.tag_features.shape[0] == len(sample_movies)
    
    # Test with empty data
    with pytest.raises(ValueError):
        cbf.fit(pd.DataFrame())
    
    # Test with missing columns
    with pytest.raises(ValueError):
        cbf.fit(pd.DataFrame({'title': ['Movie']}))

def test_get_movie_features(sample_movies):
    """Test getting movie features"""
    cbf = ContentBasedFiltering()
    cbf.fit(sample_movies)
    
    # Test valid movie ID
    genre_features, tag_features = cbf.get_movie_features(1)
    assert isinstance(genre_features, np.ndarray)
    assert isinstance(tag_features, np.ndarray)
    
    # Test non-existent movie ID
    with pytest.raises(ValueError):
        cbf.get_movie_features(999)
    
    # Test before fitting
    cbf = ContentBasedFiltering()
    with pytest.raises(ValueError):
        cbf.get_movie_features(1)

def test_similar_movies(sample_movies):
    """Test finding similar movies"""
    cbf = ContentBasedFiltering()
    cbf.fit(sample_movies)
    
    # Test getting similar movies
    similar_movies = cbf.get_similar_movies(1, n=2)
    assert len(similar_movies) == 2
    assert all(isinstance(m, tuple) and len(m) == 2 for m in similar_movies)
    assert all(0 <= score <= 1 for _, score in similar_movies)
    
    # Test with non-existent movie ID
    with pytest.raises(ValueError):
        cbf.get_similar_movies(999)
    
    # Test before fitting
    cbf = ContentBasedFiltering()
    with pytest.raises(ValueError):
        cbf.get_similar_movies(1)

def test_movie_recommendations(sample_movies):
    """Test movie recommendations"""
    cbf = ContentBasedFiltering()
    cbf.fit(sample_movies)
    
    # Test with valid ratings
    user_ratings = {1: 5.0, 2: 4.0}  # User likes drama/crime movies
    recs = cbf.recommend_movies(user_ratings, n_recommendations=2)
    assert len(recs) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    assert all(movie_id not in user_ratings for movie_id, _ in recs)
    
    # Test with empty ratings
    with pytest.raises(ValueError):
        cbf.recommend_movies({})
    
    # Test before fitting
    cbf = ContentBasedFiltering()
    with pytest.raises(ValueError):
        cbf.recommend_movies({1: 5.0})

def test_edge_cases(sample_movies):
    """Test edge cases and error handling"""
    cbf = ContentBasedFiltering()
    
    # Test with missing genres
    movies_no_genres = sample_movies.copy()
    movies_no_genres.loc[0, 'genres'] = ''
    cbf.fit(movies_no_genres)
    
    # Should still work with missing genres
    similar_movies = cbf.get_similar_movies(1, n=2)
    assert len(similar_movies) == 2
    
    # Test with missing tags
    movies_no_tags = sample_movies.copy()
    movies_no_tags['tags'] = ''
    cbf.fit(movies_no_tags)
    
    # Should still work with missing tags
    similar_movies = cbf.get_similar_movies(1, n=2)
    assert len(similar_movies) == 2

if __name__ == "__main__":
    logger.info("Starting Sprint 3 tests...")
    
    try:
        test_content_based_initialization()
        test_feature_processing(sample_movies())
        test_model_fitting(sample_movies())
        test_get_movie_features(sample_movies())
        test_similar_movies(sample_movies())
        test_movie_recommendations(sample_movies())
        test_edge_cases(sample_movies())
        logger.info("All Sprint 3 tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 