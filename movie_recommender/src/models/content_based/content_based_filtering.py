import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class ContentBasedFiltering:
    """
    Implements content-based filtering for movie recommendations based on movie features.
    
    This class provides methods for:
    1. Processing movie features (genres, tags)
    2. Computing movie similarities based on features
    3. Generating personalized recommendations based on user preferences
    
    The implementation uses TF-IDF for text features and one-hot encoding for categorical
    features, combined with cosine similarity for finding similar movies.
    
    Attributes:
        genre_features (np.ndarray): One-hot encoded genre features
        tag_features (np.ndarray): TF-IDF vectors for movie tags
        movie_ids (np.ndarray): Array of movie IDs
        genre_encoder (MultiLabelBinarizer): Encoder for genre features
        tag_vectorizer (TfidfVectorizer): Vectorizer for tag features
        logger (logging.Logger): Logger instance for tracking operations
    
    Example:
        >>> cbf = ContentBasedFiltering()
        >>> cbf.fit(movies_data)
        >>> similar_movies = cbf.get_similar_movies(movie_id=42, n=5)
    """
    
    def __init__(self):
        """Initialize the content-based filtering model."""
        self.genre_features = None
        self.tag_features = None
        self.movie_ids = None
        self.genre_encoder = MultiLabelBinarizer()
        self.tag_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger = logging.getLogger(__name__)
        
    def _process_genres(self, genres_data: pd.Series) -> np.ndarray:
        """
        Process movie genres into one-hot encoded features.
        
        Args:
            genres_data (pd.Series): Series of genre lists
            
        Returns:
            np.ndarray: One-hot encoded genre features
        """
        # Convert genre strings to lists
        genre_lists = genres_data.apply(lambda x: x.split('|') if isinstance(x, str) else [])
        return self.genre_encoder.fit_transform(genre_lists)
        
    def _process_tags(self, tags_data: pd.Series) -> np.ndarray:
        """
        Process movie tags into TF-IDF features.
        
        Args:
            tags_data (pd.Series): Series of movie tags/descriptions
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        # If all tags are empty, return zero vectors
        if tags_data.fillna('').str.strip().eq('').all():
            return np.zeros((len(tags_data), 1))
            
        return self.tag_vectorizer.fit_transform(tags_data.fillna('')).toarray()
        
    def fit(self, movies_data: pd.DataFrame) -> None:
        """
        Process movie features and prepare the model for making recommendations.
        
        Args:
            movies_data (pd.DataFrame): DataFrame containing movie information
                Must include 'movieId', 'genres', and optionally 'tags' columns
                
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            if movies_data.empty:
                raise ValueError("Input data cannot be empty")
                
            required_columns = ['movieId', 'genres']
            if not all(col in movies_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Store movie IDs
            self.movie_ids = movies_data['movieId'].values
            
            # Process genres
            self.genre_features = self._process_genres(movies_data['genres'])
            
            # Process tags if available
            if 'tags' in movies_data.columns:
                self.tag_features = self._process_tags(movies_data['tags'])
            else:
                self.tag_features = np.zeros((len(movies_data), 1))
            
            self.logger.info("Successfully processed movie features")
            
        except Exception as e:
            self.logger.error(f"Error in processing movie features: {str(e)}")
            raise
            
    def get_movie_features(self, movie_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get processed features for a specific movie.
        
        Args:
            movie_id (int): ID of the movie
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Genre features and tag features
            
        Raises:
            ValueError: If movie_id is not found or model is not fitted
        """
        try:
            if self.movie_ids is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            movie_idx = np.where(self.movie_ids == movie_id)[0]
            if len(movie_idx) == 0:
                raise ValueError(f"Movie ID {movie_id} not found")
                
            idx = movie_idx[0]
            return self.genre_features[idx], self.tag_features[idx]
            
        except Exception as e:
            self.logger.error(f"Error in getting movie features: {str(e)}")
            raise
            
    def get_similar_movies(self, movie_id: int, n: int = 5) -> List[Tuple[int, float]]:
        """
        Find movies similar to a given movie based on content features.
        
        Args:
            movie_id (int): ID of the reference movie
            n (int): Number of similar movies to return
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, similarity_score) tuples
            
        Raises:
            ValueError: If movie_id is not found or model is not fitted
        """
        try:
            if self.movie_ids is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Get movie features
            movie_idx = np.where(self.movie_ids == movie_id)[0]
            if len(movie_idx) == 0:
                raise ValueError(f"Movie ID {movie_id} not found")
                
            idx = movie_idx[0]
            
            # Compute similarities
            genre_similarities = cosine_similarity(
                self.genre_features[idx].reshape(1, -1),
                self.genre_features
            )[0]
            
            tag_similarities = cosine_similarity(
                self.tag_features[idx].reshape(1, -1),
                self.tag_features
            )[0]
            
            # Combine similarities (weighted average)
            similarities = 0.7 * genre_similarities + 0.3 * tag_similarities
            
            # Get top similar movies
            similar_indices = np.argsort(similarities)[::-1][1:n+1]  # Exclude self
            similar_movies = [
                (self.movie_ids[i], float(similarities[i]))
                for i in similar_indices
            ]
            
            return similar_movies
            
        except Exception as e:
            self.logger.error(f"Error in finding similar movies: {str(e)}")
            raise
            
    def recommend_movies(
        self,
        user_ratings: Dict[int, float],
        n_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies based on user's rating history.
        
        This method finds movies similar to those the user has rated highly,
        weighing the similarities by the user's ratings to generate personalized
        recommendations.
        
        Args:
            user_ratings (Dict[int, float]): Dictionary of {movie_id: rating}
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, score) tuples
            
        Raises:
            ValueError: If no valid ratings are provided or model is not fitted
        """
        try:
            if not user_ratings:
                raise ValueError("No user ratings provided")
                
            if self.movie_ids is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Initialize recommendation scores
            scores = np.zeros(len(self.movie_ids))
            
            # Get recommendations based on each rated movie
            for movie_id, rating in user_ratings.items():
                try:
                    # Find similar movies
                    similar_movies = self.get_similar_movies(
                        movie_id,
                        n=len(self.movie_ids)
                    )
                    
                    # Add weighted similarities to scores
                    for similar_id, similarity in similar_movies:
                        idx = np.where(self.movie_ids == similar_id)[0][0]
                        scores[idx] += similarity * (rating / 5.0)  # Normalize rating
                        
                except ValueError:
                    # Skip if movie not found
                    continue
            
            # Exclude already rated movies
            rated_indices = [
                np.where(self.movie_ids == movie_id)[0][0]
                for movie_id in user_ratings.keys()
                if movie_id in self.movie_ids
            ]
            scores[rated_indices] = -1
            
            # Get top recommendations
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            recommendations = [
                (self.movie_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in generating recommendations: {str(e)}")
            raise 