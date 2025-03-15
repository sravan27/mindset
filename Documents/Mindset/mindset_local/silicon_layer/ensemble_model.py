"""
Ensemble Model Implementation for MINDSET Silicon Layer
Implements stacked ensemble models for robust prediction of news metrics.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.silicon_layer.ensemble_model')

class StackedEnsembleModel:
    """
    Stacked ensemble model for predicting news metrics.
    
    This class implements a stacked ensemble approach combining multiple
    base models and a meta-model to make robust predictions.
    """
    
    def __init__(
        self,
        feature_cols: List[str] = None,
        target_col: str = "political_influence",
        model_dir: str = None,
        random_state: int = 42
    ):
        """
        Initialize the stacked ensemble model.
        
        Args:
            feature_cols: List of feature column names for training
            target_col: Name of the target column to predict
            model_dir: Directory to save/load models
            random_state: Random seed for reproducibility
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_dir = model_dir
        self.random_state = random_state
        
        # Initialize model components
        self.preprocessor = StandardScaler()
        self.base_models = []
        self.meta_model = None
        self.is_fitted = False
        
        # Set up model directory
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    
    def _create_base_models(self) -> List[Tuple[str, BaseEstimator]]:
        """Create the base models for the ensemble."""
        models = [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )),
            ('gbm', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ))
        ]
        return models
    
    def _create_meta_model(self) -> BaseEstimator:
        """Create the meta-model for the ensemble."""
        return LinearRegression()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StackedEnsembleModel':
        """
        Fit the stacked ensemble model.
        
        Args:
            X: Features dataframe
            y: Target series (if None, will use X[self.target_col])
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ensemble model for {self.target_col}")
        
        # Extract features and target if needed
        features = X[self.feature_cols] if self.feature_cols else X.drop(columns=[self.target_col])
        target = y if y is not None else X[self.target_col]
        
        # Split data for stacking
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=self.random_state
        )
        
        # Preprocess features
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val)
        
        # Create and fit base models
        self.base_models = self._create_base_models()
        base_model_predictions = []
        
        for name, model in self.base_models:
            logger.info(f"Fitting base model: {name}")
            model.fit(X_train_scaled, y_train)
            
            # Generate predictions for meta-model training
            preds = model.predict(X_val_scaled)
            base_model_predictions.append(preds)
            
            # Evaluate base model
            mse = mean_squared_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            logger.info(f"Base model {name} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Create meta-features for training the meta-model
        meta_features = np.column_stack(base_model_predictions)
        
        # Create and fit meta-model
        self.meta_model = self._create_meta_model()
        logger.info("Fitting meta-model")
        self.meta_model.fit(meta_features, y_val)
        
        # Final evaluation
        meta_preds = self.meta_model.predict(meta_features)
        meta_mse = mean_squared_error(y_val, meta_preds)
        meta_r2 = r2_score(y_val, meta_preds)
        logger.info(f"Meta-model - MSE: {meta_mse:.4f}, R²: {meta_r2:.4f}")
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the stacked ensemble model.
        
        Args:
            X: Features dataframe
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Extract features
        features = X[self.feature_cols] if self.feature_cols else X
        
        # Preprocess features
        X_scaled = self.preprocessor.transform(features)
        
        # Generate base model predictions
        base_model_predictions = []
        for _, model in self.base_models:
            preds = model.predict(X_scaled)
            base_model_predictions.append(preds)
        
        # Stack predictions for meta-model
        meta_features = np.column_stack(base_model_predictions)
        
        # Generate final predictions with meta-model
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the base models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Only Random Forest and Gradient Boosting models have feature_importance_
        importance_models = [model for name, model in self.base_models 
                            if hasattr(model, 'feature_importances_')]
        
        if not importance_models:
            return {}
        
        # Average feature importance across models
        feature_importances = np.zeros(len(self.feature_cols))
        for model in importance_models:
            feature_importances += model.feature_importances_
        
        feature_importances /= len(importance_models)
        
        # Create dictionary mapping feature names to importance
        importance_dict = dict(zip(self.feature_cols, feature_importances))
        
        # Sort by importance (descending)
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (optional)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Determine filepath
        if filepath is None:
            if self.model_dir is None:
                raise ValueError("No filepath or model_dir specified")
            filepath = os.path.join(self.model_dir, f"{self.target_col}_model.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model to disk
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'target_col': self.target_col,
                'preprocessor': self.preprocessor,
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'StackedEnsembleModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls(
            feature_cols=model_data['feature_cols'],
            target_col=model_data['target_col'],
            model_dir=os.path.dirname(filepath)
        )
        
        # Load model components
        instance.preprocessor = model_data['preprocessor']
        instance.base_models = model_data['base_models']
        instance.meta_model = model_data['meta_model']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class MetricEnsembleTrainer:
    """
    Trainer for multiple metric ensemble models.
    
    This class handles training multiple ensemble models for different metrics
    (political influence, rhetoric intensity, information depth).
    """
    
    def __init__(
        self,
        base_dir: str,
        feature_cols: List[str] = None,
        random_state: int = 42
    ):
        """
        Initialize the ensemble trainer.
        
        Args:
            base_dir: Base directory for model storage
            feature_cols: List of feature column names to use
            random_state: Random seed for reproducibility
        """
        self.base_dir = Path(base_dir)
        self.feature_cols = feature_cols
        self.random_state = random_state
        
        # Create model directory
        self.model_dir = self.base_dir / "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set up metric models
        self.models = {
            "political_influence": None,
            "rhetoric_intensity": None,
            "information_depth": None
        }
        
        # Set up logging
        self.logger = logger
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for training.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
        
        # If feature columns are not provided, infer them
        if self.feature_cols is None:
            # Exclude target columns and non-numeric columns
            exclude_cols = list(self.models.keys()) + ['news_id', 'information_depth_category']
            self.feature_cols = [col for col in df.columns 
                                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            
            self.logger.info(f"Inferred {len(self.feature_cols)} feature columns")
        
        return df
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Train models for all metrics.
        
        Args:
            data: Training data with features and targets
            
        Returns:
            Dictionary mapping metric names to saved model paths
        """
        # Preprocess data
        df = self._preprocess_data(data)
        
        # Train a model for each metric
        model_paths = {}
        for metric in self.models.keys():
            self.logger.info(f"Training model for {metric}")
            
            # Create and train model
            model = StackedEnsembleModel(
                feature_cols=self.feature_cols,
                target_col=metric,
                model_dir=str(self.model_dir),
                random_state=self.random_state
            )
            
            model.fit(df)
            
            # Save model
            model_path = model.save()
            model_paths[metric] = model_path
            
            # Store model reference
            self.models[metric] = model
        
        self.logger.info("All models trained and saved successfully")
        return model_paths
    
    def load_models(self) -> Dict[str, StackedEnsembleModel]:
        """
        Load all trained models from disk.
        
        Returns:
            Dictionary mapping metric names to loaded models
        """
        for metric in self.models.keys():
            model_path = self.model_dir / f"{metric}_model.pkl"
            
            if os.path.exists(model_path):
                self.models[metric] = StackedEnsembleModel.load(model_path)
                self.logger.info(f"Loaded model for {metric}")
            else:
                self.logger.warning(f"No saved model found for {metric}")
        
        return self.models
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with all models.
        
        Args:
            data: Test data with features
            
        Returns:
            DataFrame with predictions for all metrics
        """
        # Preprocess data
        df = self._preprocess_data(data)
        
        # Make predictions for each metric
        predictions = {}
        for metric, model in self.models.items():
            if model is None or not model.is_fitted:
                self.logger.warning(f"No trained model for {metric}. Skipping predictions.")
                continue
            
            self.logger.info(f"Making predictions for {metric}")
            predictions[metric] = model.predict(df[self.feature_cols])
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Add information_depth_category based on information_depth
        if 'information_depth' in pred_df.columns:
            pred_df['information_depth_category'] = pd.cut(
                pred_df['information_depth'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=['Overview', 'Analysis', 'In-depth'],
                include_lowest=True
            )
        
        return pred_df
    
    def export_model_metadata(self) -> str:
        """
        Export model metadata and feature importance.
        
        Returns:
            Path to the exported metadata file
        """
        metadata = {
            "models": {},
            "feature_importance": {}
        }
        
        for metric, model in self.models.items():
            if model is None or not model.is_fitted:
                continue
            
            # Add model metadata
            metadata["models"][metric] = {
                "feature_count": len(model.feature_cols),
                "model_path": str(self.model_dir / f"{metric}_model.pkl")
            }
            
            # Add feature importance
            metadata["feature_importance"][metric] = model.get_feature_importance()
        
        # Save metadata
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model metadata exported to {metadata_path}")
        return str(metadata_path)