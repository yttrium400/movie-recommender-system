from pathlib import Path
from typing import Dict, Any
import yaml
import os
from dotenv import load_dotenv

class Config:
    """Configuration manager for the movie recommender system."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables
        load_dotenv()
        
        # Base paths
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data"
        self.models_path = self.base_path / "models"
        self.logs_path = self.base_path / "logs"
        
        # Create necessary directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.model_params = {
            "collaborative": {
                "n_factors": 100,
                "n_epochs": 20,
                "lr_all": 0.005,
                "reg_all": 0.02
            },
            "content_based": {
                "min_df": 0.01,
                "max_df": 0.95,
                "ngram_range": [1, 2]  # Changed from tuple to list for YAML compatibility
            },
            "matrix_factorization": {
                "n_components": 50,
                "max_iter": 100,
                "learning_rate": 0.001
            }
        }
        
        # API settings
        self.api_settings = {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": True
        }
        
        # Streamlit settings
        self.streamlit_settings = {
            "page_title": "Movie Recommender",
            "page_icon": "🎬",
            "layout": "wide"
        }
        
        # Data paths
        self.data_paths = {
            'movies_file': 'data/movies.csv',
            'ratings_file': 'data/ratings.csv',
            'model_dir': 'saved_models'
        }
        
        # Preprocessing settings
        self.preprocessing = {
            'train_size': 0.7,
            'validation_size': 0.15,
            'test_size': 0.15,
            'random_state': 42
        }
        
        # Logging settings
        self.logging = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/recommender.log'
        }
        
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model type.
        
        Args:
            model_type (str): Type of model ('collaborative', 'content_based', or 'matrix_factorization')
            
        Returns:
            Dict[str, Any]: Model parameters
        """
        params = self.model_params.get(model_type, {})
        # Convert list back to tuple for ngram_range if it exists
        if model_type == "content_based" and "ngram_range" in params:
            params["ngram_range"] = tuple(params["ngram_range"])
        return params
        
    def get_data_path(self) -> Path:
        """Get the path to the data directory."""
        return self.data_path
        
    def get_models_path(self) -> Path:
        """Get the path to the models directory."""
        return self.models_path
        
    def get_logs_path(self) -> Path:
        """Get the path to the logs directory."""
        return self.logs_path
        
    def save_config(self, config_path: str = "config.yaml"):
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path (str): Path to save the configuration file
        """
        config_dict = {
            "model_params": self.model_params,
            "api_settings": self.api_settings,
            "streamlit_settings": self.streamlit_settings
        }
        
        with open(self.base_path / config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    @classmethod
    def load_config(cls, config_path: str = "config.yaml") -> 'Config':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            Config: Configuration instance
        """
        config = cls()
        config_file = config.base_path / config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            config.model_params = config_dict.get('model_params', config.model_params)
            config.api_settings = config_dict.get('api_settings', config.api_settings)
            config.streamlit_settings = config_dict.get('streamlit_settings', config.streamlit_settings)
            
        return config

# Create a global config instance
config = Config()
