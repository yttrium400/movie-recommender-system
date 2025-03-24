import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import setup_logger

class ContentBasedFiltering:
    """
    Implements content-based filtering for movie recommendations.
    
    This class uses movie features (genres, descriptions, etc.) to find similar movies
    and generate recommendations. It employs TF-IDF for feature extraction and
    cosine similarity for finding similar items.
    
    Attributes:
        movies_data (pd.DataFrame): Movie metadata
        tfidf_matrix (np.ndarray): TF-IDF feature matrix
        vectorizer (TfidfVectorizer): TF-IDF vectorizer
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(self):
        """Initialize the content-based filtering model."""
        self.movies_data = None
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger = setup_logger(__name__)
        
    def fit(self, movies_data: pd.DataFrame) -> None:
        """
        Fit the content-based filtering model.
        
        Args:
            movies_data (pd.DataFrame): Movie metadata including genres
            
        Raises:
            ValueError: If input data is invalid
        """
        if movies_data.empty:
            raise ValueError("Movies data cannot be empty")
            
        try:
            self.movies_data = movies_data
            
            # Combine features into a single text field
            content_features = movies_data['genres'].fillna('')
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(content_features)
            
            self.logger.info("Successfully fitted content-based filtering model")
            
        except Exception as e:
            self.logger.error(f"Error fitting content-based model: {str(e)}")
            raise
            
    def recommend_movies(
        self,
        user_ratings: Dict[int, float],
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Generate movie recommendations based on user's rating history.
        
        Args:
            user_ratings (Dict[int, float]): Dictionary of user's movie ratings
            n_recommendations (int): Number of recommendations to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, score) tuples
            
        Raises:
            ValueError: If parameters are invalid
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model must be fitted before generating recommendations")
            
        try:
            # Get indices of rated movies
            rated_indices = []
            for movie_id in user_ratings:
                try:
                    idx = self.movies_data.index[
                        self.movies_data['movieId'] == movie_id
                    ].tolist()[0]
                    rated_indices.append(idx)
                except IndexError:
                    continue
                    
            if not rated_indices:
                self.logger.warning("No valid rated movies found")
                return []
                
            # Calculate similarity scores for all movies
            similarity_scores = np.zeros(len(self.movies_data))
            for idx in rated_indices:
                # Get similarity scores for this movie
                movie_similarities = cosine_similarity(
                    self.tfidf_matrix[idx],
                    self.tfidf_matrix
                ).flatten()
                
                # Weight by user rating
                movie_id = self.movies_data.iloc[idx]['movieId']
                rating_weight = user_ratings[movie_id]
                similarity_scores += movie_similarities * rating_weight
                
            # Normalize scores
            if len(rated_indices) > 0:
                similarity_scores /= len(rated_indices)
                
            # Create movie ID to score mapping
            movie_scores = list(enumerate(similarity_scores))
            
            # Filter by minimum similarity and remove rated movies
            filtered_scores = [
                (self.movies_data.iloc[idx]['movieId'], score)
                for idx, score in movie_scores
                if score >= min_similarity and
                self.movies_data.iloc[idx]['movieId'] not in user_ratings
            ]
            
            # Sort by score and return top N
            recommendations = sorted(
                filtered_scores,
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            self.logger.info(
                f"Generated {len(recommendations)} content-based recommendations"
            )
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating content-based recommendations: {str(e)}")
            raise 