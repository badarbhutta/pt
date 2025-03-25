# app/ml/models.py
"""
Consolidated API for all model types.
Provides a clean interface for using different model types for period prediction.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union, Type
from enum import Enum, auto
import pandas as pd
import numpy as np
from datetime import datetime

from app.ml.model_implementations import (
    BaseModel,
    ARIMAModel,
    RandomForestModel,
    GradientBoostingModel
)
from app.core.config import settings

logger = logging.getLogger("period_tracker.models")

class ModelType(Enum):
    """Enum for available model types."""
    ARIMA = auto()
    RANDOM_FOREST = auto()
    GRADIENT_BOOSTING = auto()
    ENSEMBLE = auto()
    
    @classmethod
    def from_string(cls, model_type_str: str) -> "ModelType":
        """Convert string to ModelType enum."""
        model_type_map = {
            "arima": cls.ARIMA,
            "random_forest": cls.RANDOM_FOREST,
            "randomforest": cls.RANDOM_FOREST,
            "gradient_boosting": cls.GRADIENT_BOOSTING,
            "gradientboosting": cls.GRADIENT_BOOSTING,
            "ensemble": cls.ENSEMBLE
        }
        
        model_type_str = model_type_str.lower()
        if model_type_str not in model_type_map:
            raise ValueError(f"Unknown model type: {model_type_str}")
        
        return model_type_map[model_type_str]


class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(model_type: Union[str, ModelType], **kwargs) -> BaseModel:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Instance of the requested model type
            
        Raises:
            ValueError: If model type is not supported
        """
        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)
        
        if model_type == ModelType.ARIMA:
            return ARIMAModel(**kwargs)
        
        elif model_type == ModelType.RANDOM_FOREST:
            return RandomForestModel(**kwargs)
        
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingModel(**kwargs)
        
        elif model_type == ModelType.ENSEMBLE:
            raise NotImplementedError("Ensemble models not implemented yet")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    async def load_model(model_id: str, model_type: Optional[Union[str, ModelType]] = None) -> BaseModel:
        """
        Load an existing model.
        
        Args:
            model_id: ID of the model to load
            model_type: Type of model (optional, inferred from metadata if not provided)
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model type cannot be determined
        """
        models_directory = settings.MODEL_STORAGE_PATH
        model_path = os.path.join(models_directory, f"{model_id}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_type:
            # If model type is provided, use it to determine the class
            if isinstance(model_type, str):
                model_type = ModelType.from_string(model_type)
                
            model_class = None
            if model_type == ModelType.ARIMA:
                model_class = ARIMAModel
            elif model_type == ModelType.RANDOM_FOREST:
                model_class = RandomForestModel
            elif model_type == ModelType.GRADIENT_BOOSTING:
                model_class = GradientBoostingModel
                
            if model_class:
                return await model_class.load(model_path)
        
        # Try to infer model type from metadata file
        metadata_path = model_path.replace(".pkl", "_meta.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            model_type_str = metadata.get("model_type", "").lower()
            
            if "arima" in model_type_str:
                return await ARIMAModel.load(model_path)
            elif "randomforest" in model_type_str:
                return await RandomForestModel.load(model_path)
            elif "gradientboosting" in model_type_str:
                return await GradientBoostingModel.load(model_path)
        
        # If we can't determine the model type, raise an error
        raise ValueError(f"Could not determine model type for {model_id}")


class CyclePredictionModel:
    """High-level interface for cycle prediction models."""
    
    def __init__(self):
        """Initialize the cycle prediction model."""
        self.models_directory = settings.MODEL_STORAGE_PATH
        self.active_models = {}
        self._load_active_models()
    
    def _load_active_models(self):
        """Load active models from disk."""
        try:
            # For each model type, try to find the most recently used model
            if not os.path.exists(self.models_directory):
                return
                
            import json
            
            # Find all metadata files
            meta_files = [f for f in os.listdir(self.models_directory) if f.endswith("_meta.json")]
            
            # Group by model type
            model_types = {
                ModelType.ARIMA: [],
                ModelType.RANDOM_FOREST: [],
                ModelType.GRADIENT_BOOSTING: [],
                ModelType.ENSEMBLE: []
            }
            
            for file in meta_files:
                try:
                    with open(os.path.join(self.models_directory, file), "r") as f:
                        metadata = json.load(f)
                    
                    model_id = metadata.get("model_id")
                    model_type_str = metadata.get("model_type", "").lower()
                    last_used = metadata.get("last_used", metadata.get("created_at"))
                    
                    if "arima" in model_type_str:
                        model_types[ModelType.ARIMA].append((model_id, last_used))
                    elif "randomforest" in model_type_str:
                        model_types[ModelType.RANDOM_FOREST].append((model_id, last_used))
                    elif "gradientboosting" in model_type_str:
                        model_types[ModelType.GRADIENT_BOOSTING].append((model_id, last_used))
                    else:
                        model_types[ModelType.ENSEMBLE].append((model_id, last_used))
                
                except Exception as e:
                    logger.error(f"Error loading model metadata from {file}: {str(e)}")
            
            # For each model type, get the most recently used model
            for model_type, models in model_types.items():
                if models:
                    # Sort by last_used (most recent first)
                    models.sort(key=lambda x: x[1] if x[1] else "", reverse=True)
                    self.active_models[model_type] = models[0][0]
        
        except Exception as e:
            logger.error(f"Error loading active models: {str(e)}")
    
    async def get_active_model(self, model_type: Union[str, ModelType]) -> Optional[BaseModel]:
        """
        Get the active model of a specific type.
        
        Args:
            model_type: Type of model to get
            
        Returns:
            Model instance, or None if no active model of that type
        """
        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)
        
        model_id = self.active_models.get(model_type)
        if not model_id:
            return None
        
        try:
            return await ModelFactory.load_model(model_id, model_type)
        except Exception as e:
            logger.error(f"Error loading active model {model_id}: {str(e)}")
            return None
    
    async def predict_cycle_length(self, cycle_history: List[int], features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict the next cycle length.
        
        Args:
            cycle_history: List of previous cycle lengths
            features: Additional features for non-ARIMA models
            
        Returns:
            Prediction results
        """
        results = {}
        predictions = []
        confidences = []
        
        # Try ARIMA model
        arima_model = await self.get_active_model(ModelType.ARIMA)
        if arima_model:
            try:
                arima_result = await arima_model.predict(np.array(cycle_history))
                if arima_result.get("status") == "success":
                    arima_pred = arima_result["predictions"][0]  # First prediction
                    arima_conf = arima_result["confidence"]
                    predictions.append(arima_pred)
                    confidences.append(arima_conf)
                    results["arima"] = {
                        "prediction": arima_pred,
                        "confidence": arima_conf
                    }
            except Exception as e:
                logger.error(f"ARIMA prediction error: {str(e)}")
        
        # Try Random Forest model
        if features:
            rf_model = await self.get_active_model(ModelType.RANDOM_FOREST)
            if rf_model:
                try:
                    features_df = pd.DataFrame([features])
                    rf_result = await rf_model.predict(features_df)
                    if rf_result.get("status") == "success":
                        rf_pred = rf_result["predictions"][0]  # First prediction
                        rf_conf = rf_result["confidence"]
                        predictions.append(rf_pred)
                        confidences.append(rf_conf)
                        results["random_forest"] = {
                            "prediction": rf_pred,
                            "confidence": rf_conf
                        }
                except Exception as e:
                    logger.error(f"Random Forest prediction error: {str(e)}")
            
            # Try Gradient Boosting model
            gb_model = await self.get_active_model(ModelType.GRADIENT_BOOSTING)
            if gb_model:
                try:
                    features_df = pd.DataFrame([features])
                    gb_result = await gb_model.predict(features_df)
                    if gb_result.get("status") == "success":
                        gb_pred = gb_result["predictions"][0]  # First prediction
                        gb_conf = gb_result["confidence"]
                        predictions.append(gb_pred)
                        confidences.append(gb_conf)
                        results["gradient_boosting"] = {
                            "prediction": gb_pred,
                            "confidence": gb_conf
                        }
                except Exception as e:
                    logger.error(f"Gradient Boosting prediction error: {str(e)}")
        
        # Compute weighted average if we have predictions
        if predictions:
            total_confidence = sum(confidences)
            if total_confidence > 0:
                # Weight each prediction by its confidence
                weights = [conf/total_confidence for conf in confidences]
                weighted_prediction = sum(p * w for p, w in zip(predictions, weights))
                
                # Round to nearest integer (cycle lengths are in days)
                final_prediction = round(weighted_prediction)
                
                # Average confidence
                avg_confidence = sum(confidences) / len(confidences)
            else:
                # If all confidences are 0, use simple average
                final_prediction = round(sum(predictions) / len(predictions))
                avg_confidence = 0.5  # Default confidence
        else:
            # Default if no models succeeded
            final_prediction = 28  # Default cycle length
            avg_confidence = 0.5   # Default confidence
        
        # Return combined results
        return {
            "prediction": final_prediction,
            "confidence": avg_confidence,
            "model_contributions": results
        }
    
    async def predict_fertility_window(self, next_period_date, cycle_length: int) -> Dict[str, Any]:
        """
        Predict fertility window based on next period date and cycle length.
        
        Args:
            next_period_date: Datetime object for next period start
            cycle_length: Predicted cycle length
            
        Returns:
            Fertility window details
        """
        # Standard calculation: ovulation around 14 days before next period
        if isinstance(next_period_date, str):
            next_period_date = datetime.fromisoformat(next_period_date)
            
        # Import timedelta here since we're using it
        from datetime import timedelta
        
        # Calculate ovulation date (approximately 14 days before next period)
        ovulation_date = next_period_date - timedelta(days=14)
        
        # Fertile window is typically 5 days before ovulation to 1-2 days after
        fertility_start = ovulation_date - timedelta(days=5)
        fertility_end = ovulation_date + timedelta(days=1)
        
        return {
            "ovulation_date": ovulation_date.isoformat(),
            "fertility_window": {
                "start": fertility_start.isoformat(),
                "end": fertility_end.isoformat()
            }
        }
    
    async def set_active_model(self, model_id: str, model_type: Union[str, ModelType]) -> bool:
        """
        Set a model as active for a specific model type.
        
        Args:
            model_id: Model ID to set as active
            model_type: Type of model
            
        Returns:
            Success status
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType.from_string(model_type)
            except ValueError:
                return False
        
        try:
            # Verify model exists
            models_directory = settings.MODEL_STORAGE_PATH
            model_path = os.path.join(models_directory, f"{model_id}.pkl")
            
            if not os.path.exists(model_path):
                return False
                
            # Set as active
            self.active_models[model_type] = model_id
            return True
            
        except Exception as e:
            logger.error(f"Error setting active model: {str(e)}")
            return False


# For backward compatibility with existing code
def get_model_by_type(model_type: str) -> Type[BaseModel]:
    """
    Get model class by type string.
    
    Args:
        model_type: Type of model as string
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == "arima":
        return ARIMAModel
    elif model_type in ["random_forest", "randomforest"]:
        return RandomForestModel
    elif model_type in ["gradient_boosting", "gradientboosting"]:
        return GradientBoostingModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Module-level CyclePredictionModel instance for convenience
cycle_prediction_model = CyclePredictionModel()