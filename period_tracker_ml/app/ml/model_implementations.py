# app/ml/model_implementations.py
"""Model implementations for period prediction."""
import os
import uuid
import json
import numpy as np
import pandas as pd
import logging
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("period_tracker.models")

class BaseModel:
    """Base class for all ML models."""
    
    def __init__(self):
        """Initialize the model."""
        self.model_id = f"model_{str(uuid.uuid4())[:8]}"
        self.model_type = self.__class__.__name__
        self.created_at = datetime.now().isoformat()
        self.last_used = None
        self.metrics = {}
        self.model = None
    
    async def train(self, X, y=None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Features (for most models) or time series (for ARIMA)
            y: Target values (optional, not used for ARIMA)
            
        Returns:
            Dictionary with training results
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    async def predict(self, X, **kwargs) -> Dict[str, Any]:
        """
        Make predictions.
        
        Args:
            X: Features or time series
            
        Returns:
            Dictionary with predictions
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    async def save(self, directory: str) -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model
        """
        os.makedirs(directory, exist_ok=True)
        
        # Update last used timestamp
        self.last_used = datetime.now().isoformat()
        
        # Save the model data
        model_path = os.path.join(directory, f"{self.model_id}.pkl")
        metadata_path = os.path.join(directory, f"{self.model_id}_meta.json")
        
        # Create metadata
        metadata = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "metrics": self.metrics
        }
        
        # Save serialized model
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_id": self.model_id,
                "model_type": self.model_type
            }, f)
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model {self.model_id} saved to {model_path}")
        return model_path
    
    @classmethod
    async def load(cls, model_path: str):
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Model instance
        """
        # Load serialized model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls()
        instance.model = model_data["model"]
        instance.model_id = model_data["model_id"]
        
        # Update last used timestamp
        instance.last_used = datetime.now().isoformat()
        
        # Update metadata file
        metadata_path = model_path.replace(".pkl", "_meta.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Update metadata
            metadata["last_used"] = instance.last_used
            
            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        return instance


class ARIMAModel(BaseModel):
    """ARIMA model for time series prediction."""
    
    async def train(self, data: np.ndarray, p=1, d=1, q=0, **kwargs) -> Dict[str, Any]:
        """
        Train ARIMA model on cycle length data.
        
        Args:
            data: Array of cycle lengths
            p: AR order
            d: Integration order
            q: MA order
            
        Returns:
            Dictionary with training results
        """
        try:
            # Ensure data is a numpy array
            data = np.array(data)
            
            # Create and fit ARIMA model
            model = sm.tsa.ARIMA(data, order=(p, d, q))
            self.model = model.fit()
            
            # Make in-sample predictions for evaluation
            predictions = self.model.predict(start=1, end=len(data)-1)
            
            # Calculate metrics
            actual = data[1:] if len(predictions) == len(data) - 1 else data
            if len(predictions) != len(actual):
                actual = actual[-len(predictions):]
                
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            
            # Store metrics
            self.metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "training_samples": len(data),
                "p": p,
                "d": d,
                "q": q
            }
            
            return {
                "status": "success",
                "metrics": self.metrics,
                "model_id": self.model_id
            }
        
        except Exception as e:
            logger.exception(f"Error training ARIMA model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def predict(self, data: np.ndarray, steps=1, **kwargs) -> Dict[str, Any]:
        """
        Make predictions with ARIMA model.
        
        Args:
            data: Recent cycle length data
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }
        
        try:
            # Update last used timestamp
            self.last_used = datetime.now().isoformat()
            
            # Make forecast
            forecast = self.model.forecast(steps=steps)
            
            # Round to nearest integer (cycle lengths are in days)
            predictions = [round(float(x)) for x in forecast]
            
            # Return results
            return {
                "status": "success",
                "predictions": predictions,
                "confidence": 1.0 - (self.metrics.get("mae", 3) / 28.0)  # Simple confidence heuristic
            }
        
        except Exception as e:
            logger.exception(f"Error making predictions with ARIMA model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


class RandomForestModel(BaseModel):
    """Random Forest model for cycle prediction."""
    
    async def train(self, X: pd.DataFrame, y: np.ndarray, 
                   n_estimators=100, max_depth=None, random_state=42, **kwargs) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X: Features DataFrame
            y: Target values array
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training results
        """
        try:
            # Store feature names
            self.features = list(X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            
            # Create and train model
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage of predictions within 1 and 2 days
            within_1_day = np.mean(np.abs(y_test - y_pred) <= 1) * 100
            within_2_days = np.mean(np.abs(y_test - y_pred) <= 2) * 100
            
            # Get feature importance
            feature_importance = dict(zip(self.features, self.model.feature_importances_))
            
            # Store metrics
            self.metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "within_1_day_percent": float(within_1_day),
                "within_2_days_percent": float(within_2_days),
                "feature_importance": feature_importance,
                "training_samples": len(X),
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth is not None else "None"
            }
            
            return {
                "status": "success",
                "metrics": self.metrics,
                "model_id": self.model_id
            }
        
        except Exception as e:
            logger.exception(f"Error training Random Forest model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Make predictions with Random Forest model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }
        
        try:
            # Update last used timestamp
            self.last_used = datetime.now().isoformat()
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Round to nearest integer
            predictions = [round(float(x)) for x in predictions]
            
            # Simple confidence based on MAE
            confidence = 1.0 - (self.metrics.get("mae", 3) / 28.0)
            
            # Return results
            return {
                "status": "success",
                "predictions": predictions,
                "confidence": confidence
            }
        
        except Exception as e:
            logger.exception(f"Error making predictions with Random Forest model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for cycle prediction."""
    
    async def train(self, X: pd.DataFrame, y: np.ndarray, 
                   n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs) -> Dict[str, Any]:
        """
        Train Gradient Boosting model.
        
        Args:
            X: Features DataFrame
            y: Target values array
            n_estimators: Number of boosting stages
            learning_rate: Contribution of each tree
            max_depth: Maximum tree depth
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training results
        """
        try:
            # Store feature names
            self.features = list(X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            
            # Create and train model
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage of predictions within 1 and 2 days
            within_1_day = np.mean(np.abs(y_test - y_pred) <= 1) * 100
            within_2_days = np.mean(np.abs(y_test - y_pred) <= 2) * 100
            
            # Get feature importance
            feature_importance = dict(zip(self.features, self.model.feature_importances_))
            
            # Store metrics
            self.metrics = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "within_1_day_percent": float(within_1_day),
                "within_2_days_percent": float(within_2_days),
                "feature_importance": feature_importance,
                "training_samples": len(X),
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            }
            
            return {
                "status": "success",
                "metrics": self.metrics,
                "model_id": self.model_id
            }
        
        except Exception as e:
            logger.exception(f"Error training Gradient Boosting model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Make predictions with Gradient Boosting model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }
        
        try:
            # Update last used timestamp
            self.last_used = datetime.now().isoformat()
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Round to nearest integer
            predictions = [round(float(x)) for x in predictions]
            
            # Simple confidence based on MAE
            confidence = 1.0 - (self.metrics.get("mae", 3) / 28.0)
            
            # Return results
            return {
                "status": "success",
                "predictions": predictions,
                "confidence": confidence
            }
        
        except Exception as e:
            logger.exception(f"Error making predictions with Gradient Boosting model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


class PredictionService:
    """Service for making predictions about periods, fertility, and ovulation."""
    
    def __init__(self):
        """Initialize prediction service."""
        self.models_directory = "models"
        self.default_cycle_length = 28
        self.default_period_length = 5
        self.default_confidence = 0.75
        
        # Map of active model IDs
        self.active_models = {
            "arima": None,
            "random_forest": None,
            "gradient_boosting": None
        }
        
        # Ensure models directory exists
        os.makedirs(self.models_directory, exist_ok=True)
        
    async def predict_next_period(self, user_id: int, cycle_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict the next period start date.
        
        Args:
            user_id: User ID
            cycle_history: List of previous cycles
            
        Returns:
            Period prediction result
        """
        try:
            # Check if we have enough history for a prediction
            if not cycle_history or len(cycle_history) < 2:
                # Return a default prediction
                today = datetime.now()
                next_period = today + timedelta(days=self.default_cycle_length - 5)
                
                return {
                    "user_id": user_id,
                    "next_period_start": next_period.isoformat(),
                    "predicted_cycle_length": self.default_cycle_length,
                    "days_until_next_period": (next_period - today).days,
                    "fertility_window": {
                        "start": (next_period - timedelta(days=18)).isoformat(),
                        "end": (next_period - timedelta(days=11)).isoformat()
                    },
                    "ovulation_date": (next_period - timedelta(days=14)).isoformat(),
                    "confidence": self.default_confidence
                }
            
            # Get cycle lengths from history
            cycle_lengths = []
            for i in range(1, len(cycle_history)):
                start_curr = datetime.fromisoformat(cycle_history[i]["start_date"])
                start_prev = datetime.fromisoformat(cycle_history[i-1]["start_date"])
                cycle_lengths.append((start_curr - start_prev).days)
            
            # Use weighted average if we have enough cycles
            if len(cycle_lengths) >= 3:
                # More weight to recent cycles
                weights = [0.5, 0.3, 0.2]
                if len(cycle_lengths) > 3:
                    weights = [0.5, 0.3, 0.2] + [0.0] * (len(cycle_lengths) - 3)
                    weights = weights[-len(cycle_lengths):]
                
                predicted_length = round(sum(w * l for w, l in zip(weights, cycle_lengths[-len(weights):])))
            else:
                # Simple average otherwise
                predicted_length = round(sum(cycle_lengths) / len(cycle_lengths))
            
            # Get the last period start date
            last_period_start = datetime.fromisoformat(cycle_history[-1]["start_date"])
            
            # Calculate next period date
            next_period = last_period_start + timedelta(days=predicted_length)
            
            # Calculate days until next period
            today = datetime.now()
            days_until = (next_period - today).days
            
            # Calculate fertility window (typically 14 days before next period, +/- 3 days)
            ovulation_date = next_period - timedelta(days=14)
            fertility_window_start = ovulation_date - timedelta(days=4)
            fertility_window_end = ovulation_date + timedelta(days=1)
            
            # Return prediction
            return {
                "user_id": user_id,
                "next_period_start": next_period.isoformat(),
                "predicted_cycle_length": predicted_length,
                "days_until_next_period": days_until,
                "fertility_window": {
                    "start": fertility_window_start.isoformat(),
                    "end": fertility_window_end.isoformat()
                },
                "ovulation_date": ovulation_date.isoformat(),
                "confidence": 0.7 + min(0.2, 0.025 * len(cycle_lengths))  # Confidence increases with more history
            }
        
        except Exception as e:
            logger.exception(f"Error predicting next period: {str(e)}")
            
            # Return a default prediction on error
            today = datetime.now()
            next_period = today + timedelta(days=self.default_cycle_length - 5)
            
            return {
                "user_id": user_id,
                "next_period_start": next_period.isoformat(),
                "predicted_cycle_length": self.default_cycle_length,
                "days_until_next_period": (next_period - today).days,
                "fertility_window": {
                    "start": (next_period - timedelta(days=18)).isoformat(),
                    "end": (next_period - timedelta(days=11)).isoformat()
                },
                "ovulation_date": (next_period - timedelta(days=14)).isoformat(),
                "confidence": self.default_confidence
            }
    
    async def predict_fertility(self, user_id: int, cycle_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict fertility status and window.
        
        Args:
            user_id: User ID
            cycle_history: List of previous cycles
            
        Returns:
            Fertility prediction result
        """
        try:
            # Start by getting period prediction
            period_prediction = await self.predict_next_period(user_id, cycle_history)
            
            # Extract fertility window
            fertility_start = datetime.fromisoformat(period_prediction["fertility_window"]["start"])
            fertility_end = datetime.fromisoformat(period_prediction["fertility_window"]["end"])
            ovulation_date = datetime.fromisoformat(period_prediction["ovulation_date"])
            
            # Determine current cycle day
            today = datetime.now()
            last_period_start = datetime.now()
            if cycle_history:
                last_period_start = datetime.fromisoformat(cycle_history[-1]["start_date"])
            
            current_cycle_day = (today - last_period_start).days + 1
            
            # Determine if currently in fertile window
            is_fertile_now = fertility_start <= today <= fertility_end
            
            return {
                "user_id": user_id,
                "current_cycle_day": current_cycle_day,
                "fertility_window_start": fertility_start.isoformat(),
                "fertility_window_end": fertility_end.isoformat(),
                "is_fertile_now": is_fertile_now,
                "ovulation_date": ovulation_date.isoformat(),
                "confidence": period_prediction["confidence"] * 0.95  # Slightly lower confidence for fertility
            }
        
        except Exception as e:
            logger.exception(f"Error predicting fertility: {str(e)}")
            
            # Return a default prediction
            today = datetime.now()
            ovulation_date = today + timedelta(days=14)
            fertility_start = today + timedelta(days=10)
            fertility_end = today + timedelta(days=16)
            
            return {
                "user_id": user_id,
                "current_cycle_day": 1,
                "fertility_window_start": fertility_start.isoformat(),
                "fertility_window_end": fertility_end.isoformat(),
                "is_fertile_now": False,
                "ovulation_date": ovulation_date.isoformat(),
                "confidence": self.default_confidence * 0.8
            }
    
    async def predict_ovulation(self, user_id: int, cycle_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict ovulation date.
        
        Args:
            user_id: User ID
            cycle_history: List of previous cycles
            
        Returns:
            Ovulation prediction result
        """
        try:
            # Start by getting period prediction
            period_prediction = await self.predict_next_period(user_id, cycle_history)
            
            # Extract ovulation date
            ovulation_date = datetime.fromisoformat(period_prediction["ovulation_date"])
            
            # Calculate days to ovulation
            today = datetime.now()
            days_to_ovulation = (ovulation_date - today).days
            
            return {
                "user_id": user_id,
                "ovulation_date": ovulation_date.isoformat(),
                "days_to_ovulation": days_to_ovulation,
                "confidence": period_prediction["confidence"] * 0.9  # Slightly lower confidence for ovulation
            }
        
        except Exception as e:
            logger.exception(f"Error predicting ovulation: {str(e)}")
            
            # Return a default prediction
            today = datetime.now()
            ovulation_date = today + timedelta(days=14)
            
            return {
                "user_id": user_id,
                "ovulation_date": ovulation_date.isoformat(),
                "days_to_ovulation": 14,
                "confidence": self.default_confidence * 0.8
            }
    
    async def add_user_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback on predictions.
        
        Args:
            feedback_data: User feedback data
            
        Returns:
            Processing result
        """
        try:
            # Store feedback (in a real app, this would go to a database)
            logger.info(f"Received feedback: {feedback_data}")
            
            return {
                "status": "success",
                "message": "Feedback recorded successfully",
                "feedback_id": str(uuid.uuid4())[:8]
            }
        
        except Exception as e:
            logger.exception(f"Error processing feedback: {str(e)}")
            
            return {
                "status": "error",
                "message": f"Error processing feedback: {str(e)}"
            }