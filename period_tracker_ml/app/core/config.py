# app/core/config.py
import os
from typing import List, Optional, Union
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Period Tracker ML API"
    
    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    # Uses SQLite by default, can be overridden with environment variables
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./data/period_tracker.db"
    
    # Model storage
    MODEL_STORAGE_PATH: str = "./models"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    # API settings
    MAX_PREDICTION_HISTORY: int = 10  # Maximum number of predictions to keep per user
    
    # ML settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence for predictions
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()