# app/ml/base_model.py
import os
import json
import pickle
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the period tracking application.
    
    This class defines the standard interface for all prediction models, while
    allowing each model implementation to use different algorithms under the hood.
    """
    
    def __init__(self):
        """Initialize the base model with a unique ID."""
        self.model_id = f"model_{str(uuid.uuid4())[:8]}"
        self.model = None
        self.features = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "metrics": {},
        }
    
    @abstractmethod
    async def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model with provided data.
        
        Args:
            data: Training data in a format appropriate for the model
            **kwargs: Additional model-specific training parameters
            
        Returns:
            Dict containing training metrics and status
        """
        pass
    
    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the trained model.
        
        Args:
            features: Dictionary of features for making prediction
            
        Returns:
            Dict containing prediction results and confidence
        """
        pass
    
    async def save(self, directory: str = "models") -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model file
        """
        os.makedirs(directory, exist_ok=True)
        
        # Create model data dictionary
        model_data = {
            "model": self.model,
            "features": self.features,
            "metadata": self.metadata,
            "model_type": self.__class__.__name__
        }
        
        # Save model file
        model_path = os.path.join(directory, f"{self.model_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        # Update metadata file
        metadata_path = os.path.join(directory, f"{self.model_id}_meta.json")
        metadata = {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.metadata["created_at"],
            "last_used": datetime.now().isoformat(),
            "features": self.features,
            "metrics": self.metadata.get("metrics", {}),
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return model_path
    
    @classmethod
    async def load(cls, model_id: str, directory: str = "models") -> 'BaseModel':
        """
        Load a saved model from disk.
        
        Args:
            model_id: ID of the model to load
            directory: Directory where the model is saved
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{model_id}.pkl")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found in {directory}")
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance of the right model type
        model_type = model_data.get("model_type")
        
        if model_type == "ARIMAModel":
            from app.ml.model_implementations import ARIMAModel
            instance = ARIMAModel()
        elif model_type == "RandomForestModel":
            from app.ml.model_implementations import RandomForestModel
            instance = RandomForestModel()
        elif model_type == "GradientBoostingModel":
            from app.ml.model_implementations import GradientBoostingModel
            instance = GradientBoostingModel()
        else:
            # Default to base implementation
            instance = cls()
        
        # Load model attributes
        instance.model_id = model_id
        instance.model = model_data["model"]
        instance.features = model_data["features"]
        instance.metadata = model_data["metadata"]
        
        # Update last used timestamp
        instance.metadata["last_used"] = datetime.now().isoformat()
        
        return instance