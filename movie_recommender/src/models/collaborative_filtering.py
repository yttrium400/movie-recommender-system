import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import logging

class CollaborativeFiltering:
    """
    Implements both user-based and item-based collaborative filtering for movie recommendations.
    
    This class provides methods for:
    1. Computing user-user and item-item similarity matrices
    2. Predicting ratings for user-movie pairs
    3. Generating personalized movie recommendations
    
    The implementation uses cosine similarity for computing similarities between users and items.
    
    Attributes:
        k_neighbors (int): Number of neighbors to consider for predictions
        user_similarity_matrix (np.ndarray): Matrix of similarities between users
        item_similarity_matrix (np.ndarray): Matrix of similarities between items
        user_movie_matrix (pd.DataFrame): User-movie rating matrix
        logger (logging.Logger): Logger instance for tracking operations
    
    Example:
        >>> cf = CollaborativeFiltering(k_neighbors=5)
        >>> cf.fit(user_movie_matrix)
        >>> recommendations = cf.recommend_movies(user_id=42, n_recommendations=5)
    """
    
    def __init__(self, k_neighbors: int = 5):
        """
        Initialize the collaborative filtering model.
        
        Args:
            k_neighbors (int): Number of neighbors to use for predictions. A larger
                number may provide more stable predictions but will be slower.
        """
        self.k_neighbors = k_neighbors
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_movie_matrix = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        Fit the model by computing similarity matrices.
        
        This method computes both user-user and item-item similarity matrices using
        cosine similarity. These matrices are used for making predictions and
        recommendations.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix where rows
                represent users and columns represent movies. Values are ratings.
        
        Raises:
            ValueError: If the input matrix is empty or contains invalid values.
        """
        try:
            self.user_movie_matrix = user_movie_matrix
            
            # Compute user similarity matrix
            self.user_similarity_matrix = cosine_similarity(user_movie_matrix)
            
            # Compute item similarity matrix
            self.item_similarity_matrix = cosine_similarity(user_movie_matrix.T)
            
            self.logger.info("Successfully computed similarity matrices")
            
        except Exception as e:
            self.logger.error(f"Error in fitting collaborative filtering model: {str(e)}")
            raise
            
    def predict_user_based(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using user-based collaborative filtering.
        
        This method finds similar users and uses their ratings to predict the
        rating for the target user-movie pair. The prediction is a weighted average
        of ratings from similar users.
        
        Args:
            user_id (int): ID of the user
            movie_id (int): ID of the movie
            
        Returns:
            float: Predicted rating in the range [0, 5]
            
        Raises:
            ValueError: If the model hasn't been fitted or if IDs are invalid.
        """
        try:
            if self.user_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            # Get user's row index
            user_idx = self.user_movie_matrix.index.get_loc(user_id)
            movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
            
            # Get similar users
            similar_users = np.argsort(self.user_similarity_matrix[user_idx])[-self.k_neighbors-1:-1]
            
            # Get ratings of similar users for this movie
            similar_ratings = self.user_movie_matrix.iloc[similar_users, movie_idx]
            similarities = self.user_similarity_matrix[user_idx, similar_users]
            
            # Compute weighted average
            if np.sum(similarities) == 0:
                return 0.0
            
            predicted_rating = float(np.sum(similar_ratings * similarities) / np.sum(similarities))
            return predicted_rating
            
        except Exception as e:
            self.logger.error(f"Error in user-based prediction: {str(e)}")
            raise
            
    def predict_item_based(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using item-based collaborative filtering.
        
        This method finds similar items and uses the user's ratings on those items
        to predict the rating for the target movie. The prediction is a weighted
        average of ratings based on item similarities.
        
        Args:
            user_id (int): ID of the user
            movie_id (int): ID of the movie
            
        Returns:
            float: Predicted rating in the range [0, 5]
            
        Raises:
            ValueError: If the model hasn't been fitted or if IDs are invalid.
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            # Get indices
            user_idx = self.user_movie_matrix.index.get_loc(user_id)
            movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
            
            # Get similar items
            similar_items = np.argsort(self.item_similarity_matrix[movie_idx])[-self.k_neighbors-1:-1]
            
            # Get user's ratings for similar items
            user_ratings = self.user_movie_matrix.iloc[user_idx, similar_items]
            similarities = self.item_similarity_matrix[movie_idx, similar_items]
            
            # Compute weighted average
            if np.sum(similarities) == 0:
                return 0.0
            
            predicted_rating = float(np.sum(user_ratings * similarities) / np.sum(similarities))
            return predicted_rating
            
        except Exception as e:
            self.logger.error(f"Error in item-based prediction: {str(e)}")
            raise
            
    def recommend_movies(self, user_id: int, n_recommendations: int = 5, method: str = 'user') -> List[Tuple[int, float]]:
        """
        Recommend movies for a user.
        
        This method finds movies that the user hasn't rated yet and predicts their
        ratings using either user-based or item-based collaborative filtering. The
        movies are then sorted by predicted rating to generate recommendations.
        
        Args:
            user_id (int): ID of the user
            n_recommendations (int): Number of recommendations to return
            method (str): 'user' for user-based or 'item' for item-based filtering
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples,
                sorted by predicted rating in descending order
            
        Raises:
            ValueError: If the method is invalid or if user_id is invalid
        """
        try:
            if method not in ['user', 'item']:
                raise ValueError("Method must be either 'user' or 'item'")
                
            # Get movies user hasn't rated
            user_ratings = self.user_movie_matrix.loc[user_id]
            unwatched_movies = user_ratings[user_ratings == 0].index
            
            # Predict ratings for unwatched movies
            predictions = []
            for movie_id in unwatched_movies:
                if method == 'user':
                    rating = self.predict_user_based(user_id, movie_id)
                else:
                    rating = self.predict_item_based(user_id, movie_id)
                predictions.append((movie_id, rating))
            
            # Sort by predicted rating and return top n
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            self.logger.error(f"Error in movie recommendations: {str(e)}")
            raise 