import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

class MovieDataLoader:
    """
    Class to handle loading and basic processing of movie recommendation datasets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
    def download_datasets(self) -> None:
        """
        Download the MovieLens dataset if not present.
        Uses the small MovieLens dataset (100k) for development.
        """
        try:
            import requests
            
            # MovieLens dataset base URL
            base_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            
            # Create data directory if it doesn't exist
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and extract the dataset
            zip_path = self.data_dir / "ml-latest-small.zip"
            if not (self.data_dir / "movies.csv").exists():
                self.logger.info("Downloading MovieLens dataset...")
                response = requests.get(base_url)
                response.raise_for_status()
                
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Move files from the extracted directory to data directory
                extracted_dir = self.data_dir / "ml-latest-small"
                for file in extracted_dir.glob("*.csv"):
                    file.rename(self.data_dir / file.name)
                
                # Clean up
                import shutil
                shutil.rmtree(extracted_dir)
                zip_path.unlink()
                
                self.logger.info("Successfully downloaded and extracted dataset")
                    
        except Exception as e:
            self.logger.error(f"Error downloading datasets: {str(e)}")
            raise
            
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the movies and ratings datasets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: movies and ratings dataframes
        """
        try:
            # Download datasets if they don't exist
            self.download_datasets()
            
            # Load datasets
            movies_df = pd.read_csv(self.data_dir / "movies.csv")
            ratings_df = pd.read_csv(self.data_dir / "ratings.csv")
            
            self.logger.info("Successfully loaded movies and ratings datasets")
            return movies_df, ratings_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def get_user_movie_matrix(self, ratings_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a user-movie matrix from ratings data.
        
        Args:
            ratings_df (Optional[pd.DataFrame]): Ratings dataframe. If None, loads from file.
            
        Returns:
            pd.DataFrame: User-movie matrix with users as rows and movies as columns
        """
        try:
            if ratings_df is None:
                _, ratings_df = self.load_data()
                
            user_movie_matrix = ratings_df.pivot(
                index='userId',
                columns='movieId',
                values='rating'
            ).fillna(0)
            
            self.logger.info("Successfully created user-movie matrix")
            return user_movie_matrix
            
        except Exception as e:
            self.logger.error(f"Error creating user-movie matrix: {str(e)}")
            raise
