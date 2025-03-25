# app/ml/training.py
"""Training service for ML models."""
import os
import uuid
import json
import asyncio
import logging
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import BackgroundTasks

from app.ml.model_implementations import (
    BaseModel,
    ARIMAModel,
    RandomForestModel,
    GradientBoostingModel
)
from app.core.config import settings

logger = logging.getLogger("period_tracker.training")

class TrainingService:
    """Service for training and managing ML models."""
    
    def __init__(self):
        """Initialize training service."""
        self.models_directory = settings.MODEL_STORAGE_PATH
        os.makedirs(self.models_directory, exist_ok=True)
        
        # Dictionary to track training jobs
        self.training_jobs = {}
        
        # Dictionary of active models
        self.active_models = {
            "arima": None,
            "random_forest": None,
            "gradient_boosting": None,
            "ensemble": None
        }
        
        # Load any existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models from disk."""
        try:
            # Check for model files in the directory
            if not os.path.exists(self.models_directory):
                return
            
            model_files = [f for f in os.listdir(self.models_directory) if f.endswith(".pkl")]
            
            for file in model_files:
                try:
                    model_path = os.path.join(self.models_directory, file)
                    
                    # Load model information from metadata file
                    meta_path = model_path.replace(".pkl", "_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            metadata = json.load(f)
                        
                        logger.info(f"Found existing model: {metadata.get('model_id')} of type {metadata.get('model_type')}")
                        
                        # Mark as active if it's the most recent model of its type
                        # (In a real app, we'd have a more sophisticated active model selection)
                        model_type_lower = metadata.get('model_type', "").lower()
                        if model_type_lower == "arimamodel":
                            if not self.active_models["arima"]:
                                self.active_models["arima"] = metadata.get('model_id')
                        elif model_type_lower == "randomforestmodel":
                            if not self.active_models["random_forest"]:
                                self.active_models["random_forest"] = metadata.get('model_id')
                        elif model_type_lower == "gradientboostingmodel":
                            if not self.active_models["gradient_boosting"]:
                                self.active_models["gradient_boosting"] = metadata.get('model_id')
                
                except Exception as e:
                    logger.error(f"Error loading model {file}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error loading existing models: {str(e)}")
    
    async def start_training(
        self, 
        model_type: str,
        training_params: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        background_tasks: BackgroundTasks = None
    ) -> str:
        """
        Start a new training job.
        
        Args:
            model_type: Type of model to train
            training_params: Parameters for training
            user_id: User ID (for personalized models)
            background_tasks: FastAPI background tasks
            
        Returns:
            Training job ID
        """
        # Generate a unique training ID
        training_id = f"training_{str(uuid.uuid4())[:8]}"
        
        # Create job status
        self.training_jobs[training_id] = {
            "training_id": training_id,
            "model_type": model_type,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "progress": 0,
            "message": "Training job created",
            "model_id": None,
            "metrics": {},
            "training_params": training_params or {}
        }
        
        # Start training in background
        if background_tasks:
            background_tasks.add_task(
                self._run_training_job,
                training_id=training_id,
                model_type=model_type,
                training_params=training_params or {},
                user_id=user_id
            )
        else:
            # For testing or direct calls without FastAPI
            asyncio.create_task(
                self._run_training_job(
                    training_id=training_id,
                    model_type=model_type,
                    training_params=training_params or {},
                    user_id=user_id
                )
            )
        
        return training_id
    
    async def _run_training_job(
        self, 
        training_id: str,
        model_type: str,
        training_params: Dict[str, Any],
        user_id: Optional[int] = None
    ):
        """
        Run a training job in the background.
        
        Args:
            training_id: Training job ID
            model_type: Type of model to train
            training_params: Parameters for training
            user_id: User ID (for personalized models)
        """
        try:
            # Update job status
            self.training_jobs[training_id]["status"] = "running"
            self.training_jobs[training_id]["updated_at"] = datetime.now().isoformat()
            self.training_jobs[training_id]["message"] = "Training started"
            
            # For demo purposes, we'll simulate training with fake data
            # In a real application, we would fetch real user data from the database
            
            # Create dummy data based on model type
            if model_type.lower() == "arima":
                # Time series data for ARIMA
                import numpy as np
                
                # Sample cycle lengths with some noise
                cycle_lengths = [28, 30, 29, 31, 28, 29, 30, 32, 29, 28]
                train_data = np.array(cycle_lengths)
                
                # Create and train ARIMA model
                model = ARIMAModel()
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 20
                self.training_jobs[training_id]["message"] = "Model initialized, starting training"
                
                # Get p, d, q parameters or use defaults
                p = training_params.get("p", 1)
                d = training_params.get("d", 1)
                q = training_params.get("q", 0)
                
                # Train the model
                result = await model.train(train_data, p=p, d=d, q=q)
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 80
                self.training_jobs[training_id]["message"] = "Model trained, saving"
                
                # Save model if training was successful
                if result["status"] == "success":
                    model_path = await model.save(self.models_directory)
                    model_id = model.model_id
                    
                    # Update job with model info
                    self.training_jobs[training_id]["model_id"] = model_id
                    self.training_jobs[training_id]["metrics"] = model.metrics
                    
                    # Set as active ARIMA model
                    self.active_models["arima"] = model_id
                    
                    # Update job status
                    self.training_jobs[training_id]["status"] = "completed"
                    self.training_jobs[training_id]["message"] = f"Model trained and saved as {model_id}"
                    self.training_jobs[training_id]["progress"] = 100
                else:
                    # Training failed
                    self.training_jobs[training_id]["status"] = "failed"
                    self.training_jobs[training_id]["message"] = f"Training failed: {result.get('message', 'Unknown error')}"
                
            elif model_type.lower() in ["random_forest", "randomforest"]:
                # Tabular data for Random Forest
                import pandas as pd
                import numpy as np
                
                # Create sample features
                X = pd.DataFrame({
                    "previous_cycle_length": [28, 30, 29, 31, 28, 29, 30, 32, 29, 28],
                    "average_cycle_length": [29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5],
                    "cycle_length_std": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                    "user_age": [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
                    "has_regular_cycle": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "stress_level": [2, 3, 2, 1, 2, 3, 2, 1, 2, 2],
                    "exercise_level": [2, 2, 2, 2, 3, 3, 3, 3, 2, 2],
                })
                
                # Create target variable (next cycle length)
                y = np.array([29, 28, 30, 29, 31, 30, 28, 29, 30, 29])
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 20
                self.training_jobs[training_id]["message"] = "Data prepared, initializing model"
                
                # Create and train Random Forest model
                model = RandomForestModel()
                
                # Get training parameters or use defaults
                n_estimators = training_params.get("n_estimators", 100)
                max_depth = training_params.get("max_depth", None)
                
                # Train the model
                result = await model.train(X, y, n_estimators=n_estimators, max_depth=max_depth)
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 80
                self.training_jobs[training_id]["message"] = "Model trained, saving"
                
                # Save model if training was successful
                if result["status"] == "success":
                    model_path = await model.save(self.models_directory)
                    model_id = model.model_id
                    
                    # Update job with model info
                    self.training_jobs[training_id]["model_id"] = model_id
                    self.training_jobs[training_id]["metrics"] = model.metrics
                    
                    # Set as active model
                    self.active_models["random_forest"] = model_id
                    
                    # Update job status
                    self.training_jobs[training_id]["status"] = "completed"
                    self.training_jobs[training_id]["message"] = f"Model trained and saved as {model_id}"
                    self.training_jobs[training_id]["progress"] = 100
                else:
                    # Training failed
                    self.training_jobs[training_id]["status"] = "failed"
                    self.training_jobs[training_id]["message"] = f"Training failed: {result.get('message', 'Unknown error')}"
            
            elif model_type.lower() in ["gradient_boosting", "gradientboosting"]:
                # Tabular data for Gradient Boosting
                import pandas as pd
                import numpy as np
                
                # Create sample features
                X = pd.DataFrame({
                    "previous_cycle_length": [28, 30, 29, 31, 28, 29, 30, 32, 29, 28],
                    "average_cycle_length": [29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5, 29.5],
                    "cycle_length_std": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                    "user_age": [28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
                    "has_regular_cycle": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "stress_level": [2, 3, 2, 1, 2, 3, 2, 1, 2, 2],
                    "exercise_level": [2, 2, 2, 2, 3, 3, 3, 3, 2, 2],
                })
                
                # Create target variable (next cycle length)
                y = np.array([29, 28, 30, 29, 31, 30, 28, 29, 30, 29])
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 20
                self.training_jobs[training_id]["message"] = "Data prepared, initializing model"
                
                # Create and train Gradient Boosting model
                model = GradientBoostingModel()
                
                # Get training parameters or use defaults
                n_estimators = training_params.get("n_estimators", 100)
                learning_rate = training_params.get("learning_rate", 0.1)
                max_depth = training_params.get("max_depth", 3)
                
                # Train the model
                result = await model.train(
                    X, y, 
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
                
                # Update progress
                self.training_jobs[training_id]["progress"] = 80
                self.training_jobs[training_id]["message"] = "Model trained, saving"
                
                # Save model if training was successful
                if result["status"] == "success":
                    model_path = await model.save(self.models_directory)
                    model_id = model.model_id
                    
                    # Update job with model info
                    self.training_jobs[training_id]["model_id"] = model_id
                    self.training_jobs[training_id]["metrics"] = model.metrics
                    
                    # Set as active model
                    self.active_models["gradient_boosting"] = model_id
                    
                    # Update job status
                    self.training_jobs[training_id]["status"] = "completed"
                    self.training_jobs[training_id]["message"] = f"Model trained and saved as {model_id}"
                    self.training_jobs[training_id]["progress"] = 100
                else:
                    # Training failed
                    self.training_jobs[training_id]["status"] = "failed"
                    self.training_jobs[training_id]["message"] = f"Training failed: {result.get('message', 'Unknown error')}"
            
            elif model_type.lower() == "ensemble":
                # For ensemble model, we would train multiple models and combine them
                # For this demo, we'll just create a placeholder
                
                # Update status
                self.training_jobs[training_id]["status"] = "failed"
                self.training_jobs[training_id]["message"] = "Ensemble models not implemented yet"
                
            else:
                # Unsupported model type
                self.training_jobs[training_id]["status"] = "failed"
                self.training_jobs[training_id]["message"] = f"Unsupported model type: {model_type}"
            
            # Final update to timestamp
            self.training_jobs[training_id]["updated_at"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.exception(f"Error running training job: {str(e)}")
            
            # Update job status on error
            self.training_jobs[training_id]["status"] = "failed"
            self.training_jobs[training_id]["message"] = f"Error: {str(e)}"
            self.training_jobs[training_id]["updated_at"] = datetime.now().isoformat()
    
    async def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            training_id: Training job ID
            
        Returns:
            Training job status
            
        Raises:
            ValueError: If training job not found
        """
        if training_id not in self.training_jobs:
            raise ValueError(f"Training job {training_id} not found")
        
        return self.training_jobs[training_id]
    
    async def list_models(
        self, 
        model_type: Optional[str] = None,
        user_id: Optional[int] = None,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            model_type: Filter by model type
            user_id: Filter by user ID
            active_only: Only return active models
            
        Returns:
            List of model information
        """
        models = []
        
        try:
            # Check for model files in the directory
            if not os.path.exists(self.models_directory):
                return models
            
            model_files = [f for f in os.listdir(self.models_directory) if f.endswith("_meta.json")]
            
            for file in model_files:
                try:
                    # Load model metadata
                    meta_path = os.path.join(self.models_directory, file)
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Apply filters
                    model_id = metadata.get("model_id")
                    model_type_value = metadata.get("model_type", "")
                    
                    # Skip if model type doesn't match
                    if model_type and model_type.lower() not in model_type_value.lower():
                        continue
                    
                    # Skip if user ID doesn't match
                    if user_id is not None and metadata.get("user_id") != user_id:
                        continue
                    
                    # Skip if active_only and model is not active
                    if active_only and model_id not in self.active_models.values():
                        continue
                    
                    # Add active status to metadata
                    metadata["active"] = model_id in self.active_models.values()
                    
                    # Add to results
                    models.append(metadata)
                
                except Exception as e:
                    logger.error(f"Error loading model metadata {file}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return models
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information
            
        Raises:
            ValueError: If model not found
        """
        try:
            # Look for model metadata file
            meta_path = os.path.join(self.models_directory, f"{model_id}_meta.json")
            
            if not os.path.exists(meta_path):
                raise ValueError(f"Model {model_id} not found")
            
            # Load metadata
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            
            # Add active status
            metadata["active"] = model_id in self.active_models.values()
            
            return metadata
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error getting model info: {str(e)}")
            raise ValueError(f"Error retrieving model info: {str(e)}")
    
    async def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metrics
            
        Raises:
            ValueError: If model not found
        """
        # Get model info (which includes metrics)
        model_info = await self.get_model_info(model_id)
        
        # Extract and return metrics
        return {
            "model_id": model_id,
            "model_type": model_info.get("model_type"),
            "metrics": model_info.get("metrics", {}),
            "created_at": model_info.get("created_at")
        }
    
    async def activate_model(self, model_id: str) -> bool:
        """
        Activate a model for predictions.
        
        Args:
            model_id: Model ID
            
        Returns:
            Success status
            
        Raises:
            ValueError: If model not found
        """
        try:
            # Get model info
            model_info = await self.get_model_info(model_id)
            model_type = model_info.get("model_type", "").lower()
            
            # Map model type to active_models key
            if "arima" in model_type:
                self.active_models["arima"] = model_id
            elif "randomforest" in model_type:
                self.active_models["random_forest"] = model_id
            elif "gradientboosting" in model_type:
                self.active_models["gradient_boosting"] = model_id
            else:
                # Default to setting it as an ensemble model
                self.active_models["ensemble"] = model_id
            
            logger.info(f"Model {model_id} activated as {model_type}")
            return True
        
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error activating model: {str(e)}")
            return False
    
    async def get_feature_importance(self, model_id: str) -> Dict[str, Any]:
        """
        Get feature importance for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Feature importance
            
        Raises:
            ValueError: If model not found or doesn't support feature importance
        """
        try:
            # Get model info
            model_info = await self.get_model_info(model_id)
            model_type = model_info.get("model_type", "").lower()
            metrics = model_info.get("metrics", {})
            
            # Check if model supports feature importance
            if "feature_importance" not in metrics:
                raise ValueError(f"Model {model_id} does not support feature importance")
            
            # Return feature importance
            return {
                "model_id": model_id,
                "model_type": model_info.get("model_type"),
                "feature_importance": metrics.get("feature_importance", {}),
            }
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error getting feature importance: {str(e)}")
            raise ValueError(f"Error retrieving feature importance: {str(e)}")