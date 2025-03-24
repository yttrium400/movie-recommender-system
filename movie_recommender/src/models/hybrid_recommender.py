import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import MinMaxScaler
import logging
from src.utils.logger import setup_logger

from .collaborative_filtering import CollaborativeFiltering
from .content_based.content_based_filtering import ContentBasedFiltering

class HybridRecommender:
    """
    Implements a hybrid recommendation system combining collaborative and content-based filtering.
    
    This class provides methods for:
    1. Combining predictions from multiple recommendation models
    2. Weighting recommendations based on user preferences
    3. Generating personalized recommendations using multiple approaches
    
    The implementation uses a weighted combination of collaborative and content-based
    recommendations, with dynamic weight adjustment based on user behavior.
    
    Attributes:
        collaborative_model (CollaborativeFiltering): Collaborative filtering model
        content_based_model (ContentBasedFiltering): Content-based filtering model
        collab_weight (float): Weight for collaborative filtering predictions
        content_weight (float): Weight for content-based predictions
        scaler (MinMaxScaler): Scaler for normalizing prediction scores
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(
        self,
        collab_weight: float = 0.6,
        content_weight: float = 0.4
    ):
        """
        Initialize the hybrid recommender.
        
        Args:
            collab_weight (float): Initial weight for collaborative filtering
            content_weight (float): Initial weight for content-based filtering
            
        Raises:
            ValueError: If weights don't sum to 1 or are invalid
        """
        if not 0 <= collab_weight <= 1 or not 0 <= content_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
            
        if abs(collab_weight + content_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.collaborative_model = CollaborativeFiltering()
        self.content_based_model = ContentBasedFiltering()
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
    def fit(
        self,
        user_movie_matrix: pd.DataFrame,
        movies_data: pd.DataFrame
    ) -> None:
        """
        Fit both collaborative and content-based models.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix
            movies_data (pd.DataFrame): Movie metadata
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Fit collaborative filtering model
            self.collaborative_model.fit(user_movie_matrix)
            
            # Fit content-based model
            self.content_based_model.fit(movies_data)
            
            self.logger.info("Successfully fitted hybrid recommender")
            
        except Exception as e:
            self.logger.error(f"Error fitting hybrid recommender: {str(e)}")
            raise
            
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> np.ndarray:
        """
        Normalize recommendation scores to [0, 1] range.
        
        Args:
            scores (List[Tuple[int, float]]): List of (movie_id, score) tuples
            
        Returns:
            np.ndarray: Array of normalized scores
        """
        if not scores:
            return np.array([])
            
        # Extract scores and reshape for scaling
        score_values = np.array([score for _, score in scores]).reshape(-1, 1)
        
        # Handle case where all scores are the same
        if np.all(score_values == score_values[0]):
            return np.ones(len(scores))
            
        return self.scaler.fit_transform(score_values).ravel()
        
    def recommend_movies(
        self,
        user_id: int,
        user_ratings: Dict[int, float],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Generate personalized movie recommendations using both models.
        
        Args:
            user_id (int): ID of the user
            user_ratings (Dict[int, float]): Dictionary of user's movie ratings
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, score) tuples
            
        Raises:
            ValueError: If parameters are invalid
        """
        try:
            # Get recommendations from both models
            collab_recs = self.collaborative_model.recommend_movies(
                user_id,
                n_recommendations=n_recommendations * 2
            )
            
            content_recs = self.content_based_model.recommend_movies(
                user_ratings,
                n_recommendations=n_recommendations * 2
            )
            
            # Normalize scores from both models
            collab_scores = self._normalize_scores(collab_recs)
            content_scores = self._normalize_scores(content_recs)
            
            # Create dictionaries for easy lookup
            collab_dict = {
                movie_id: score * self.collab_weight
                for (movie_id, _), score in zip(collab_recs, collab_scores)
            }
            
            content_dict = {
                movie_id: score * self.content_weight
                for (movie_id, _), score in zip(content_recs, content_scores)
            }
            
            # Combine scores
            all_movies = set(collab_dict.keys()) | set(content_dict.keys())
            combined_scores = [
                (movie_id, collab_dict.get(movie_id, 0) + content_dict.get(movie_id, 0))
                for movie_id in all_movies
            ]
            
            # Sort by score and return top N
            recommendations = sorted(
                combined_scores,
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            self.logger.info(f"Generated {len(recommendations)} hybrid recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating hybrid recommendations: {str(e)}")
            raise
            
    def update_weights(
        self,
        user_ratings: Dict[int, float],
        recommendations: List[Tuple[int, float]],
        method: str = 'performance'
    ) -> None:
        """
        Update model weights based on recommendation performance.
        
        Args:
            user_ratings (Dict[int, float]): New user ratings
            recommendations (List[Tuple[int, float]]): Previous recommendations
            method (str): Weight update method ('performance' or 'adaptive')
            
        Raises:
            ValueError: If method is invalid
        """
        try:
            if method not in ['performance', 'adaptive']:
                raise ValueError("Invalid weight update method")
                
            if method == 'performance':
                # Calculate error for each model
                collab_error = self._calculate_model_error(
                    self.collaborative_model,
                    user_ratings,
                    recommendations
                )
                
                content_error = self._calculate_model_error(
                    self.content_based_model,
                    user_ratings,
                    recommendations
                )
                
                # Update weights based on inverse error
                total_error = collab_error + content_error
                if total_error > 0:
                    self.collab_weight = 1 - (collab_error / total_error)
                    self.content_weight = 1 - (content_error / total_error)
                    
            else:  # adaptive
                # Implement adaptive weight adjustment based on user behavior
                pass
                
            self.logger.info(
                f"Updated weights: collaborative={self.collab_weight:.2f}, "
                f"content-based={self.content_weight:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")
            raise
            
    def _calculate_model_error(
        self,
        model: Union[CollaborativeFiltering, ContentBasedFiltering],
        user_ratings: Dict[int, float],
        recommendations: List[Tuple[int, float]]
    ) -> float:
        """
        Calculate prediction error for a model.
        
        Args:
            model: Recommendation model
            user_ratings (Dict[int, float]): Actual user ratings
            recommendations (List[Tuple[int, float]]): Model's recommendations
            
        Returns:
            float: Mean squared error of predictions
        """
        try:
            errors = []
            for movie_id, actual_rating in user_ratings.items():
                # Find this movie in recommendations
                predicted_rating = 0.0
                for rec_movie_id, rec_rating in recommendations:
                    if rec_movie_id == movie_id:
                        predicted_rating = rec_rating
                        break
                        
                # Calculate squared error
                error = (actual_rating - predicted_rating) ** 2
                errors.append(error)
                
            # Return mean squared error
            if errors:
                mse = np.mean(errors)
                return float(mse)
            else:
                return 1.0  # Return maximum error if no predictions
                
        except Exception as e:
            self.logger.error(f"Error calculating model error: {str(e)}")
            return 1.0  # Return maximum error on failure 