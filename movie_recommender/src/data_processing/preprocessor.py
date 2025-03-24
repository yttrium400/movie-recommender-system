import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import logging

class MovieDataPreprocessor:
    """
    Class to handle preprocessing of movie data for recommendation systems.
    """
    
    def __init__(self):
        """Initialize the preprocessor with necessary components."""
        self.logger = logging.getLogger(__name__)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
        
    def process_movie_metadata(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process movie metadata including genres, title, etc.
        
        Args:
            movies_df (pd.DataFrame): Raw movies dataframe
            
        Returns:
            pd.DataFrame: Processed movies dataframe
        """
        try:
            # Create a copy to avoid modifying the original
            df = movies_df.copy()
            
            # Extract year from title and use 1900 as default for missing years
            df['year'] = df['title'].str.extract(r'\((\d{4})\)').fillna(1900).astype('int32')
            df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
            
            # Process genres
            df['genres'] = df['genres'].fillna('')
            
            # Create genre indicator columns
            genre_dummies = df['genres'].str.get_dummies(sep='|').astype('bool')
            df = pd.concat([df, genre_dummies], axis=1)
            
            self.logger.info("Successfully processed movie metadata")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing movie metadata: {str(e)}")
            raise
            
    def process_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process ratings data including normalization and handling missing values.
        
        Args:
            ratings_df (pd.DataFrame): Raw ratings dataframe
            
        Returns:
            pd.DataFrame: Processed ratings dataframe
        """
        try:
            df = ratings_df.copy()
            
            # Remove duplicates if any
            df = df.drop_duplicates()
            
            # Normalize ratings
            df['rating_normalized'] = self.scaler.fit_transform(df[['rating']])
            
            # Add timestamp-based features
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour_of_day'] = df['timestamp'].dt.hour
            
            self.logger.info("Successfully processed ratings data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing ratings data: {str(e)}")
            raise
            
    def create_content_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Create content-based features using TF-IDF on genres.
        
        Args:
            movies_df (pd.DataFrame): Processed movies dataframe
            
        Returns:
            np.ndarray: TF-IDF matrix of genre features
        """
        try:
            # Combine genres into a single text field
            genre_text = movies_df['genres'].fillna('')
            
            # Create TF-IDF features
            genre_features = self.tfidf.fit_transform(genre_text)
            
            self.logger.info("Successfully created content features")
            return genre_features
            
        except Exception as e:
            self.logger.error(f"Error creating content features: {str(e)}")
            raise
            
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the TF-IDF features.
        
        Returns:
            List[str]: List of feature names
        """
        return self.tfidf.get_feature_names_out()
        
    def prepare_data_for_training(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Prepare all data for model training.
        
        Args:
            movies_df (pd.DataFrame): Raw movies dataframe
            ratings_df (pd.DataFrame): Raw ratings dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Processed movies, ratings, and content features
        """
        try:
            # Process both dataframes
            processed_movies = self.process_movie_metadata(movies_df)
            processed_ratings = self.process_ratings(ratings_df)
            
            # Create content features
            content_features = self.create_content_features(processed_movies)
            
            self.logger.info("Successfully prepared all data for training")
            return processed_movies, processed_ratings, content_features
            
        except Exception as e:
            self.logger.error(f"Error preparing data for training: {str(e)}")
            raise

    def split_data(self, ratings_df: pd.DataFrame, method: str = 'time', train_size: float = 0.7, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the ratings data into train, validation, and test sets.
        
        Args:
            ratings_df (pd.DataFrame): Processed ratings dataframe
            method (str): 'time' for chronological split or 'random' for random split
            train_size (float): Proportion of data for training
            val_size (float): Proportion of data for validation
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
            
        Raises:
            ValueError: If method is invalid or if split proportions are invalid
        """
        try:
            if method not in ['time', 'random']:
                raise ValueError("Method must be either 'time' or 'random'")
                
            if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
                raise ValueError("Invalid split proportions")
                
            test_size = 1 - train_size - val_size
            
            if method == 'time':
                # Sort by timestamp
                df = ratings_df.sort_values('timestamp')
                
                # Calculate split indices
                train_idx = int(len(df) * train_size)
                val_idx = int(len(df) * (train_size + val_size))
                
                # Split data
                train = df.iloc[:train_idx]
                val = df.iloc[train_idx:val_idx]
                test = df.iloc[val_idx:]
                
            else:  # random split
                # Shuffle data
                df = ratings_df.sample(frac=1, random_state=42)
                
                # Calculate split indices
                train_idx = int(len(df) * train_size)
                val_idx = int(len(df) * (train_size + val_size))
                
                # Split data
                train = df.iloc[:train_idx]
                val = df.iloc[train_idx:val_idx]
                test = df.iloc[val_idx:]
            
            self.logger.info(f"Successfully split data using {method} method")
            return train, val, test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a user-movie matrix from ratings data.
        
        Args:
            ratings_df (pd.DataFrame): Processed ratings dataframe
            
        Returns:
            pd.DataFrame: User-movie matrix with users as rows and movies as columns
            
        Raises:
            ValueError: If required columns are missing
        """
        try:
            required_columns = ['userId', 'movieId', 'rating']
            if not all(col in ratings_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Create pivot table without filling NaN values
            user_movie_matrix = ratings_df.pivot(
                index='userId',
                columns='movieId',
                values='rating'
            )
            
            self.logger.info("Successfully created user-movie matrix")
            return user_movie_matrix
            
        except Exception as e:
            self.logger.error(f"Error creating user-movie matrix: {str(e)}")
            raise
