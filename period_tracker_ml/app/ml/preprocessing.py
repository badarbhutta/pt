# app/ml/preprocessing.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

class CyclePreprocessor:
    """
    Class for preprocessing menstrual cycle data for machine learning models.
    """
    
    @staticmethod
    def normalize_cycle_data(cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize cycle data by ensuring all required fields are present and in the correct format.
        
        Args:
            cycles: List of cycle dictionaries
            
        Returns:
            List of normalized cycle dictionaries
        """
        normalized_cycles = []
        
        for cycle in cycles:
            normalized_cycle = cycle.copy()
            
            # Ensure dates are datetime objects
            for date_field in ['start_date', 'end_date']:
                if date_field in normalized_cycle:
                    if isinstance(normalized_cycle[date_field], str):
                        try:
                            normalized_cycle[date_field] = datetime.fromisoformat(normalized_cycle[date_field])
                        except ValueError:
                            # If parsing fails, try common formats
                            try:
                                normalized_cycle[date_field] = datetime.strptime(normalized_cycle[date_field], "%Y-%m-%d")
                            except ValueError:
                                # Remove invalid date
                                normalized_cycle[date_field] = None
            
            # Set default values for missing fields
            if 'cycle_length' not in normalized_cycle or normalized_cycle['cycle_length'] is None:
                normalized_cycle['cycle_length'] = 28
            
            if 'period_length' not in normalized_cycle or normalized_cycle['period_length'] is None:
                normalized_cycle['period_length'] = 5
            
            # Ensure boolean fields are boolean
            for bool_field in ['has_cramps', 'has_headache', 'has_acne', 'has_fatigue', 'has_bloating', 'has_breast_tenderness']:
                if bool_field in normalized_cycle:
                    if isinstance(normalized_cycle[bool_field], str):
                        normalized_cycle[bool_field] = normalized_cycle[bool_field].lower() in ['true', 'yes', '1', 't', 'y']
                    elif not isinstance(normalized_cycle[bool_field], bool):
                        normalized_cycle[bool_field] = bool(normalized_cycle[bool_field])
                else:
                    normalized_cycle[bool_field] = False
            
            # Ensure severity fields are integers between 0-5
            for severity_field in ['cramps_severity', 'headache_severity']:
                if severity_field in normalized_cycle and normalized_cycle[severity_field] is not None:
                    try:
                        normalized_cycle[severity_field] = max(0, min(5, int(normalized_cycle[severity_field])))
                    except (ValueError, TypeError):
                        normalized_cycle[severity_field] = 0
                else:
                    normalized_cycle[severity_field] = 0
            
            normalized_cycles.append(normalized_cycle)
        
        return normalized_cycles
    
    @staticmethod
    def extract_time_series(cycles: List[Dict[str, Any]], feature: str = 'cycle_length') -> np.ndarray:
        """
        Extract a time series from cycle data for a specific feature.
        
        Args:
            cycles: List of cycle dictionaries
            feature: Feature to extract as time series
            
        Returns:
            numpy array of feature values ordered by date
        """
        # Sort cycles by date
        sorted_cycles = sorted(cycles, key=lambda x: x.get('start_date', datetime(2000, 1, 1)))
        
        # Extract feature values
        time_series = []
        for cycle in sorted_cycles:
            if feature in cycle and cycle[feature] is not None:
                time_series.append(cycle[feature])
        
        return np.array(time_series)
    
    @staticmethod
    def create_features_dataframe(cycles: List[Dict[str, Any]], window_size: int = 3) -> pd.DataFrame:
        """
        Create a features dataframe for machine learning models.
        
        Args:
            cycles: List of cycle dictionaries
            window_size: Number of previous cycles to include as features
            
        Returns:
            DataFrame of features
        """
        # Sort cycles by date
        sorted_cycles = sorted(cycles, key=lambda x: x.get('start_date', datetime(2000, 1, 1)))
        
        # Need at least window_size + 1 cycles
        if len(sorted_cycles) < window_size + 1:
            return pd.DataFrame()
        
        # Create features
        features_list = []
        for i in range(window_size, len(sorted_cycles)):
            feature_dict = {
                'user_id': sorted_cycles[i]['user_id'],
                'target_cycle_length': sorted_cycles[i]['cycle_length']
            }
            
            # Add features from previous cycles
            for j in range(1, window_size + 1):
                prev_cycle = sorted_cycles[i - j]
                feature_dict[f'prev_cycle_length_{j}'] = prev_cycle['cycle_length']
                feature_dict[f'prev_period_length_{j}'] = prev_cycle['period_length']
                
                # Add symptom features
                for symptom in ['has_cramps', 'has_headache', 'has_acne', 'has_fatigue', 'has_bloating', 'has_breast_tenderness']:
                    feature_dict[f'prev_{symptom}_{j}'] = int(prev_cycle.get(symptom, False))
                
                # Add severity features
                for severity in ['cramps_severity', 'headache_severity']:
                    feature_dict[f'prev_{severity}_{j}'] = prev_cycle.get(severity, 0)
            
            # Calculate rolling statistics
            prev_cycle_lengths = [sorted_cycles[i-j]['cycle_length'] for j in range(1, window_size + 1)]
            feature_dict['avg_prev_cycle_length'] = np.mean(prev_cycle_lengths)
            feature_dict['std_prev_cycle_length'] = np.std(prev_cycle_lengths)
            feature_dict['min_prev_cycle_length'] = np.min(prev_cycle_lengths)
            feature_dict['max_prev_cycle_length'] = np.max(prev_cycle_lengths)
            
            # Calculate trend features
            feature_dict['cycle_trend'] = prev_cycle_lengths[0] - prev_cycle_lengths[-1]
            
            # Add season/month feature
            if 'start_date' in sorted_cycles[i] and sorted_cycles[i]['start_date']:
                feature_dict['month'] = sorted_cycles[i]['start_date'].month
                feature_dict['season'] = (sorted_cycles[i]['start_date'].month % 12) // 3 + 1
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    @staticmethod
    def split_train_test(features_df: pd.DataFrame, target_col: str = 'target_cycle_length', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            features_df: DataFrame of features
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if target_col not in features_df.columns or features_df.empty:
            raise ValueError(f"Target column {target_col} not found or dataframe is empty")
        
        # Separate features and target
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def prepare_prediction_features(cycles: List[Dict[str, Any]], window_size: int = 3) -> Optional[Dict[str, Any]]:
        """
        Prepare features for making a prediction for the next cycle.
        
        Args:
            cycles: List of cycle dictionaries
            window_size: Number of previous cycles to include
            
        Returns:
            Dictionary of features for prediction, or None if not enough data
        """
        # Sort cycles by date
        sorted_cycles = sorted(cycles, key=lambda x: x.get('start_date', datetime(2000, 1, 1)), reverse=True)
        
        # Need at least window_size cycles
        if len(sorted_cycles) < window_size:
            return None
        
        # Create feature dictionary
        features = {
            'user_id': sorted_cycles[0]['user_id']
        }
        
        # Add features from previous cycles
        for j in range(window_size):
            cycle = sorted_cycles[j]
            features[f'prev_cycle_length_{j+1}'] = cycle['cycle_length']
            features[f'prev_period_length_{j+1}'] = cycle['period_length']
            
            # Add symptom features
            for symptom in ['has_cramps', 'has_headache', 'has_acne', 'has_fatigue', 'has_bloating', 'has_breast_tenderness']:
                features[f'prev_{symptom}_{j+1}'] = int(cycle.get(symptom, False))
            
            # Add severity features
            for severity in ['cramps_severity', 'headache_severity']:
                features[f'prev_{severity}_{j+1}'] = cycle.get(severity, 0)
        
        # Calculate rolling statistics
        cycle_lengths = [cycle['cycle_length'] for cycle in sorted_cycles[:window_size]]
        features['avg_prev_cycle_length'] = np.mean(cycle_lengths)
        features['std_prev_cycle_length'] = np.std(cycle_lengths)
        features['min_prev_cycle_length'] = np.min(cycle_lengths)
        features['max_prev_cycle_length'] = np.max(cycle_lengths)
        
        # Calculate trend features
        features['cycle_trend'] = cycle_lengths[0] - cycle_lengths[-1]
        
        # Add season/month feature
        if 'start_date' in sorted_cycles[0] and sorted_cycles[0]['start_date']:
            features['month'] = sorted_cycles[0]['start_date'].month
            features['season'] = (sorted_cycles[0]['start_date'].month % 12) // 3 + 1
        
        return features