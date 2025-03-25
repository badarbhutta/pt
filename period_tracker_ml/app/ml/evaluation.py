# app/ml/evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

class ModelEvaluator:
    """
    Class for evaluating menstrual cycle prediction models.
    """
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics for numerical predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage of predictions within thresholds
        within_one_day = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        within_two_days = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        within_three_days = np.mean(np.abs(y_true - y_pred) <= 3) * 100
        
        # Calculate median absolute error
        med_ae = np.median(np.abs(y_true - y_pred))
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "median_ae": float(med_ae),
            "within_1_day_percent": float(within_one_day),
            "within_2_days_percent": float(within_two_days),
            "within_3_days_percent": float(within_three_days),
            "samples": len(y_true)
        }
    
    @staticmethod
    def calculate_date_prediction_metrics(true_dates: List[datetime], pred_dates: List[datetime]) -> Dict[str, float]:
        """
        Calculate metrics for date predictions.
        
        Args:
            true_dates: List of true date values
            pred_dates: List of predicted date values
            
        Returns:
            Dict of metrics
        """
        # Convert to days difference
        days_diff = [(pred - true).days for pred, true in zip(pred_dates, true_dates)]
        
        abs_days_diff = np.abs(days_diff)
        
        # Calculate metrics
        mae = float(np.mean(abs_days_diff))
        rmse = float(np.sqrt(np.mean(np.square(days_diff))))
        med_ae = float(np.median(abs_days_diff))
        
        # Calculate percentage of predictions within thresholds
        within_one_day = np.mean(abs_days_diff <= 1) * 100
        within_two_days = np.mean(abs_days_diff <= 2) * 100
        within_three_days = np.mean(abs_days_diff <= 3) * 100
        
        return {
            "mae_days": mae,
            "rmse_days": rmse,
            "median_ae_days": med_ae,
            "within_1_day_percent": float(within_one_day),
            "within_2_days_percent": float(within_two_days),
            "within_3_days_percent": float(within_three_days),
            "samples": len(true_dates)
        }
    
    @staticmethod
    def calculate_precision_recall_for_fertility(
        true_fertile_days: List[datetime],
        pred_fertile_days: List[datetime]
    ) -> Dict[str, float]:
        """
        Calculate precision and recall for fertility window predictions.
        
        Args:
            true_fertile_days: List of true fertile days
            pred_fertile_days: List of predicted fertile days
            
        Returns:
            Dict of metrics
        """
        # Convert to set of date strings for comparison
        true_days_set = {d.strftime("%Y-%m-%d") for d in true_fertile_days}
        pred_days_set = {d.strftime("%Y-%m-%d") for d in pred_fertile_days}
        
        # Calculate intersection
        intersection = true_days_set.intersection(pred_days_set)
        
        # Calculate metrics
        precision = len(intersection) / len(pred_days_set) if pred_days_set else 0
        recall = len(intersection) / len(true_days_set) if true_days_set else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_window_size": len(true_days_set),
            "pred_window_size": len(pred_days_set),
            "overlap_days": len(intersection)
        }
    
    @staticmethod
    def cross_validate_model(
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a regression model.
        
        Args:
            model: Model object with fit and predict methods
            X: Features DataFrame
            y: Target values
            n_splits: Number of cross-validation folds
            random_state: Random seed for reproducibility
            
        Returns:
            Dict of cross-validation metrics
        """
        from sklearn.model_selection import KFold
        
        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Store metrics for each fold
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data for this fold
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = ModelEvaluator.calculate_regression_metrics(y_test, y_pred)
            metrics['fold'] = fold + 1
            
            fold_metrics.append(metrics)
        
        # Calculate average metrics across folds
        avg_metrics = {
            "avg_mae": np.mean([m['mae'] for m in fold_metrics]),
            "avg_rmse": np.mean([m['rmse'] for m in fold_metrics]),
            "avg_r2": np.mean([m['r2'] for m in fold_metrics]),
            "avg_within_1_day_percent": np.mean([m['within_1_day_percent'] for m in fold_metrics]),
            "avg_within_2_days_percent": np.mean([m['within_2_days_percent'] for m in fold_metrics]),
            "n_splits": n_splits,
            "total_samples": len(y),
            "fold_metrics": fold_metrics
        }
        
        return avg_metrics
    
    @staticmethod
    def evaluate_forecast_history(
        predicted_cycles: List[Dict[str, Any]],
        actual_cycles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate forecasting history accuracy.
        
        Args:
            predicted_cycles: List of dictionaries with predictions
            actual_cycles: List of dictionaries with actual values
            
        Returns:
            Dict of metrics over time
        """
        # Match predictions with actuals
        matched_data = []
        
        for pred in predicted_cycles:
            pred_date = datetime.fromisoformat(pred['predicted_date']) if isinstance(pred['predicted_date'], str) else pred['predicted_date']
            
            # Find corresponding actual cycle
            for actual in actual_cycles:
                actual_date = datetime.fromisoformat(actual['start_date']) if isinstance(actual['start_date'], str) else actual['start_date']
                
                # Use date proximity to match
                if abs((pred_date - actual_date).days) < 10:  # Within 10 days is considered a match
                    matched_data.append({
                        'prediction_date': pred.get('created_at'),
                        'predicted_date': pred_date,
                        'actual_date': actual_date,
                        'days_difference': (pred_date - actual_date).days,
                        'abs_days_difference': abs((pred_date - actual_date).days),
                        'cycle_id': actual.get('id'),
                        'prediction_id': pred.get('id')
                    })
                    break
        
        # Calculate metrics by month
        if matched_data:
            df = pd.DataFrame(matched_data)
            df['prediction_month'] = pd.to_datetime(df['prediction_date']).dt.to_period('M')
            
            monthly_metrics = df.groupby('prediction_month').agg(
                mean_abs_error=('abs_days_difference', 'mean'),
                within_1_day=('abs_days_difference', lambda x: (x <= 1).mean() * 100),
                within_2_days=('abs_days_difference', lambda x: (x <= 2).mean() * 100),
                count=('prediction_id', 'count')
            ).reset_index()
            
            monthly_metrics['prediction_month'] = monthly_metrics['prediction_month'].astype(str)
            
            # Overall metrics
            overall_metrics = {
                'overall_mae': float(df['abs_days_difference'].mean()),
                'overall_within_1_day_percent': float((df['abs_days_difference'] <= 1).mean() * 100),
                'overall_within_2_days_percent': float((df['abs_days_difference'] <= 2).mean() * 100),
                'overall_within_3_days_percent': float((df['abs_days_difference'] <= 3).mean() * 100),
                'total_predictions': len(df),
                'monthly_metrics': monthly_metrics.to_dict(orient='records')
            }
            
            return overall_metrics
            
        return {
            'overall_mae': None,
            'total_predictions': 0,
            'error': 'No matching predictions found'
        }

    @staticmethod
    def plot_prediction_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction Accuracy",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot prediction accuracy as actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot (if None, returns the figure)
            
        Returns:
            Figure object if save_path is None, otherwise None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # Plot predictions
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Add labels
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add metrics text
        metrics = ModelEvaluator.calculate_regression_metrics(y_true, y_pred)
        metrics_text = (
            f"MAE: {metrics['mae']:.2f}\n"
            f"RMSE: {metrics['rmse']:.2f}\n"
            f"R²: {metrics['r2']:.2f}\n"
            f"Within 1 day: {metrics['within_1_day_percent']:.1f}%\n"
            f"Within 2 days: {metrics['within_2_days_percent']:.1f}%"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        else:
            return fig

    @staticmethod
    def plot_prediction_errors(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction Errors",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot histogram of prediction errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot (if None, returns the figure)
            
        Returns:
            Figure object if save_path is None, otherwise None
        """
        errors = y_pred - y_true
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(errors, bins=range(int(min(errors))-1, int(max(errors))+2), alpha=0.7, edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Zero Error')
        
        # Add labels
        ax.set_xlabel('Prediction Error (days)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add metrics text
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        within_one_day = np.mean(np.abs(errors) <= 1) * 100
        within_two_days = np.mean(np.abs(errors) <= 2) * 100
        
        metrics_text = (
            f"Mean Error: {mean_error:.2f} days\n"
            f"Std Dev: {std_error:.2f} days\n"
            f"Within ±1 day: {within_one_day:.1f}%\n"
            f"Within ±2 days: {within_two_days:.1f}%"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        else:
            return fig