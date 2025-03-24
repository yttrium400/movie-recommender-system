import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import setup_logger

class CollaborativeFiltering:
    """
    Implements user-based collaborative filtering for movie recommendations.
    
    This class uses user-user similarity to generate recommendations based on
    ratings from similar users. It employs cosine similarity as the similarity metric
    and can handle both explicit and implicit feedback.
    
    Attributes:
        user_movie_matrix (pd.DataFrame): User-movie rating matrix
        similarity_matrix (np.ndarray): User-user similarity matrix
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(self):
        """Initialize the collaborative filtering model."""
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.logger = setup_logger(__name__)
        
    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        Fit the collaborative filtering model.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix
            
        Raises:
            ValueError: If input data is invalid
        """
        if user_movie_matrix.empty:
            raise ValueError("User-movie matrix cannot be empty")
            
        try:
            self.user_movie_matrix = user_movie_matrix
            
            # Fill NaN values with 0 for similarity calculation
            matrix_for_sim = user_movie_matrix.fillna(0)
            
            # Calculate user-user similarity matrix
            self.similarity_matrix = cosine_similarity(matrix_for_sim)
            
            self.logger.info("Successfully fitted collaborative filtering model")
            
        except Exception as e:
            self.logger.error(f"Error fitting collaborative model: {str(e)}")
            raise
            
    def recommend_movies(
        self,
        user_id: int,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id (int): ID of the user
            n_recommendations (int): Number of recommendations to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
            
        Raises:
            ValueError: If parameters are invalid
        """
        if self.user_movie_matrix is None:
            raise ValueError("Model must be fitted before generating recommendations")
            
        try:
            # Get user's index in the matrix
            user_idx = self.user_movie_matrix.index.get_loc(user_id)
            
            # Get similar users
            user_similarities = self.similarity_matrix[user_idx]
            similar_users = np.where(user_similarities >= min_similarity)[0]
            
            if len(similar_users) == 0:
                self.logger.warning(f"No similar users found for user {user_id}")
                return []
                
            # Get movies not rated by the user
            user_ratings = self.user_movie_matrix.iloc[user_idx]
            unrated_movies = user_ratings[user_ratings.isna() | (user_ratings == 0)].index
            
            if len(unrated_movies) == 0:
                self.logger.warning(f"No unrated movies found for user {user_id}")
                return []
                
            # Calculate predicted ratings
            predictions = []
            for movie_id in unrated_movies:
                # Get ratings for this movie from similar users
                movie_ratings = self.user_movie_matrix[movie_id].iloc[similar_users]
                
                # Remove NaN values
                valid_ratings = movie_ratings.dropna()
                valid_similarities = user_similarities[similar_users][~movie_ratings.isna()]
                
                # Skip if no valid ratings
                if len(valid_ratings) == 0:
                    continue
                    
                # Calculate weighted average rating
                weighted_sum = np.sum(valid_ratings * valid_similarities)
                weight_sum = np.sum(valid_similarities)
                
                if weight_sum > 0:
                    predicted_rating = weighted_sum / weight_sum
                    predictions.append((movie_id, predicted_rating))
                    
            # Sort by predicted rating and return top N
            recommendations = sorted(
                predictions,
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            self.logger.info(
                f"Generated {len(recommendations)} recommendations for user {user_id}"
            )
            return recommendations
            
        except Exception as e:
            self.logger.error(
                f"Error generating recommendations for user {user_id}: {str(e)}"
            )
            raise 