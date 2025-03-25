# app/__init__.py
"""Period Tracker ML Application."""

# Import main services
from app.ml.model_implementations import PredictionService
from app.ml.training import TrainingService

# Initialize services
prediction_service = PredictionService()
training_service = TrainingService()

__all__ = ["prediction_service", "training_service"]