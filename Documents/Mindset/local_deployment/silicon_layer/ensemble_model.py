"""
Ensemble Model module for the Silicon Layer
Implements various ensemble strategies for news metrics prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
import warnings

class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model for Silicon Layer
    
    Implements multiple ensemble strategies:
    - Voting (weighted average)
    - Stacking (meta-learner)
    - Boosting (sequential learning)
    
    Targeting:
    - Political Influence Level
    - Rhetoric Intensity Scale
    - Information Depth Score
    """
    
    def __init__(
        self,
        ensemble_type: str = 'voting',
        models: List[BaseEstimator] = None,
        weights: Optional[List[float]] = None,
        meta_model: Optional[BaseEstimator] = None,
        models_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the ensemble model
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'boosting')
            models: List of models for ensemble
            weights: Weights for voting ensemble
            meta_model: Meta-model for stacking ensemble
            models_dir: Directory to save/load models
        """
        self.ensemble_type = ensemble_type
        self.models = models or []
        self.weights = weights
        self.meta_model = meta_model
        self.models_dir = Path(models_dir) if models_dir else None
        
        # Check ensemble type
        valid_types = ['voting', 'stacking', 'boosting']
        if ensemble_type not in valid_types:
            raise ValueError(f"Ensemble type must be one of {valid_types}")
        
        # Validate weights for voting ensemble
        if ensemble_type == 'voting' and weights and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Create weights if not provided
        if ensemble_type == 'voting' and not weights:
            self.weights = [1.0 / len(models) for _ in models] if models else []
        
        # Initialize meta-model for stacking if not provided
        if ensemble_type == 'stacking' and not meta_model:
            self.meta_model = GradientBoostingRegressor()
        
        # Create directory if provided
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def add_model(self, model: BaseEstimator, weight: float = None):
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
            weight: Weight for voting ensemble
        """
        self.models.append(model)
        
        # Update weights for voting ensemble
        if self.ensemble_type == 'voting':
            if weight:
                if not self.weights:
                    self.weights = [1.0] * (len(self.models) - 1)
                self.weights.append(weight)
                
                # Normalize weights
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]
            else:
                # Equal weights
                self.weights = [1.0 / len(self.models) for _ in self.models]
    
    def fit(self, X, y, **fit_params):
        """
        Fit the ensemble model
        
        Args:
            X: Training features
            y: Target values
            **fit_params: Additional parameters for model fitting
            
        Returns:
            self
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        if self.ensemble_type == 'voting':
            # Fit each model individually
            for i, model in enumerate(self.models):
                model.fit(X, y, **fit_params)
        
        elif self.ensemble_type == 'stacking':
            # Create out-of-fold predictions for meta-model
            n_splits = fit_params.get('n_splits', 5)
            from sklearn.model_selection import KFold
            
            meta_features = np.zeros((X.shape[0], len(self.models)))
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # For each model, create out-of-fold predictions
            for i, model in enumerate(self.models):
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                    
                    # Fit on training fold
                    model_clone = joblib.clone(model)
                    model_clone.fit(X_train, y_train)
                    
                    # Predict on validation fold
                    meta_features[val_idx, i] = model_clone.predict(X_val)
                
                # Fit on entire dataset
                model.fit(X, y)
            
            # Fit meta-model on out-of-fold predictions
            self.meta_model.fit(meta_features, y)
        
        elif self.ensemble_type == 'boosting':
            # Simple residual boosting
            # Start with predictions of zeros
            residual = y.copy()
            
            # Fit each model on residual error
            for i, model in enumerate(self.models):
                model.fit(X, residual)
                predictions = model.predict(X)
                residual -= predictions
        
        return self
    
    def predict(self, X):
        """
        Generate predictions with the ensemble model
        
        Args:
            X: Features
            
        Returns:
            Predicted values
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        if self.ensemble_type == 'voting':
            # Get predictions from each model
            predictions = np.zeros((X.shape[0], len(self.models)))
            
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(X)
            
            # Weighted average
            return np.sum(predictions * np.array(self.weights), axis=1)
        
        elif self.ensemble_type == 'stacking':
            # Get predictions from each model
            meta_features = np.zeros((X.shape[0], len(self.models)))
            
            for i, model in enumerate(self.models):
                meta_features[:, i] = model.predict(X)
            
            # Meta-model prediction
            return self.meta_model.predict(meta_features)
        
        elif self.ensemble_type == 'boosting':
            # Sum predictions from all models
            predictions = np.zeros(X.shape[0])
            
            for model in self.models:
                predictions += model.predict(X)
            
            return predictions
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the ensemble model
        
        Args:
            X: Features
            y: True values
            metrics: Metrics to compute
            
        Returns:
            Dictionary of metric scores
        """
        # Default metrics
        if metrics is None:
            metrics = {
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score
            }
        
        predictions = self.predict(X)
        
        # Compute metrics
        results = {}
        for name, metric_fn in metrics.items():
            results[name] = metric_fn(y, predictions)
        
        return results
    
    def save(self, name: str = None):
        """
        Save the ensemble model
        
        Args:
            name: Name for the saved model
            
        Returns:
            Path to saved model
        """
        if not self.models_dir:
            raise ValueError("models_dir not set. Provide a directory path in constructor.")
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate name if not provided
        if name is None:
            import time
            timestamp = int(time.time())
            name = f"ensemble_{self.ensemble_type}_{timestamp}"
        
        # Create directory for this ensemble
        ensemble_dir = self.models_dir / name
        ensemble_dir.mkdir(exist_ok=True)
        
        # Save base models
        for i, model in enumerate(self.models):
            model_path = ensemble_dir / f"model_{i}.pkl"
            joblib.dump(model, model_path)
        
        # Save meta-model if stacking
        if self.ensemble_type == 'stacking' and self.meta_model:
            meta_model_path = ensemble_dir / "meta_model.pkl"
            joblib.dump(self.meta_model, meta_model_path)
        
        # Save configuration
        config = {
            'ensemble_type': self.ensemble_type,
            'models': [f"model_{i}.pkl" for i in range(len(self.models))],
            'weights': self.weights if self.ensemble_type == 'voting' else None,
            'meta_model': "meta_model.pkl" if self.ensemble_type == 'stacking' else None
        }
        
        config_path = ensemble_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return ensemble_dir
    
    @classmethod
    def load(cls, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None):
        """
        Load an ensemble model
        
        Args:
            path: Path to saved ensemble
            models_dir: Directory for models (if not the same as path)
            
        Returns:
            Loaded EnsembleModel
        """
        path = Path(path)
        
        # Load configuration
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create ensemble
        ensemble = cls(
            ensemble_type=config['ensemble_type'],
            weights=config['weights'],
            models_dir=models_dir or path.parent
        )
        
        # Load base models
        for model_file in config['models']:
            model_path = path / model_file
            model = joblib.load(model_path)
            ensemble.models.append(model)
        
        # Load meta-model if stacking
        if config['ensemble_type'] == 'stacking' and config['meta_model']:
            meta_model_path = path / config['meta_model']
            ensemble.meta_model = joblib.load(meta_model_path)
        
        return ensemble


class MetricsEnsembleModel:
    """
    Combined ensemble model for all news metrics
    
    Handles:
    - Political Influence Level
    - Rhetoric Intensity Scale
    - Information Depth Score
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the metrics ensemble model
        
        Args:
            models_dir: Directory to save/load models
        """
        self.models_dir = Path(models_dir) if models_dir else None
        
        # Create separate ensembles for each metric
        self.political_influence_model = None
        self.rhetoric_intensity_model = None
        self.information_depth_model = None
        
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def set_political_influence_model(self, model: EnsembleModel):
        """Set the political influence model"""
        self.political_influence_model = model
    
    def set_rhetoric_intensity_model(self, model: EnsembleModel):
        """Set the rhetoric intensity model"""
        self.rhetoric_intensity_model = model
    
    def set_information_depth_model(self, model: EnsembleModel):
        """Set the information depth model"""
        self.information_depth_model = model
    
    def fit(self, X, y_dict, **fit_params):
        """
        Fit all models
        
        Args:
            X: Features
            y_dict: Dictionary of targets for each metric
            **fit_params: Additional parameters for model fitting
            
        Returns:
            self
        """
        if 'political_influence' in y_dict and self.political_influence_model:
            self.political_influence_model.fit(X, y_dict['political_influence'], **fit_params)
        
        if 'rhetoric_intensity' in y_dict and self.rhetoric_intensity_model:
            self.rhetoric_intensity_model.fit(X, y_dict['rhetoric_intensity'], **fit_params)
        
        if 'information_depth' in y_dict and self.information_depth_model:
            self.information_depth_model.fit(X, y_dict['information_depth'], **fit_params)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions for all metrics
        
        Args:
            X: Features
            
        Returns:
            Dictionary of predictions for each metric
        """
        predictions = {}
        
        if self.political_influence_model:
            predictions['political_influence'] = self.political_influence_model.predict(X)
        
        if self.rhetoric_intensity_model:
            predictions['rhetoric_intensity'] = self.rhetoric_intensity_model.predict(X)
        
        if self.information_depth_model:
            predictions['information_depth'] = self.information_depth_model.predict(X)
        
        return predictions
    
    def evaluate(self, X, y_dict, metrics=None):
        """
        Evaluate all models
        
        Args:
            X: Features
            y_dict: Dictionary of targets for each metric
            metrics: Metrics to compute
            
        Returns:
            Dictionary of evaluation results for each metric
        """
        results = {}
        
        if 'political_influence' in y_dict and self.political_influence_model:
            results['political_influence'] = self.political_influence_model.evaluate(
                X, y_dict['political_influence'], metrics)
        
        if 'rhetoric_intensity' in y_dict and self.rhetoric_intensity_model:
            results['rhetoric_intensity'] = self.rhetoric_intensity_model.evaluate(
                X, y_dict['rhetoric_intensity'], metrics)
        
        if 'information_depth' in y_dict and self.information_depth_model:
            results['information_depth'] = self.information_depth_model.evaluate(
                X, y_dict['information_depth'], metrics)
        
        return results
    
    def save(self, name: str = None):
        """
        Save all models
        
        Args:
            name: Name for the saved models
            
        Returns:
            Path to saved models
        """
        if not self.models_dir:
            raise ValueError("models_dir not set. Provide a directory path in constructor.")
        
        # Generate name if not provided
        if name is None:
            import time
            timestamp = int(time.time())
            name = f"metrics_ensemble_{timestamp}"
        
        # Create directory for these models
        models_dir = self.models_dir / name
        models_dir.mkdir(exist_ok=True)
        
        # Save individual models
        saved_paths = {}
        
        if self.political_influence_model:
            path = self.political_influence_model.save("political_influence")
            saved_paths['political_influence'] = str(path.relative_to(self.models_dir))
        
        if self.rhetoric_intensity_model:
            path = self.rhetoric_intensity_model.save("rhetoric_intensity")
            saved_paths['rhetoric_intensity'] = str(path.relative_to(self.models_dir))
        
        if self.information_depth_model:
            path = self.information_depth_model.save("information_depth")
            saved_paths['information_depth'] = str(path.relative_to(self.models_dir))
        
        # Save configuration
        config = {
            'models': saved_paths
        }
        
        config_path = models_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return models_dir
    
    @classmethod
    def load(cls, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None):
        """
        Load all models
        
        Args:
            path: Path to saved models
            models_dir: Directory for models (if not the same as path)
            
        Returns:
            Loaded MetricsEnsembleModel
        """
        path = Path(path)
        models_root = models_dir or path.parent
        
        # Load configuration
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create metrics ensemble
        metrics_ensemble = cls(models_dir=models_root)
        
        # Load individual models
        if 'political_influence' in config['models']:
            model_path = models_root / config['models']['political_influence']
            metrics_ensemble.political_influence_model = EnsembleModel.load(model_path, models_root)
        
        if 'rhetoric_intensity' in config['models']:
            model_path = models_root / config['models']['rhetoric_intensity']
            metrics_ensemble.rhetoric_intensity_model = EnsembleModel.load(model_path, models_root)
        
        if 'information_depth' in config['models']:
            model_path = models_root / config['models']['information_depth']
            metrics_ensemble.information_depth_model = EnsembleModel.load(model_path, models_root)
        
        return metrics_ensemble


def create_default_metrics_ensemble(models_dir: Optional[Union[str, Path]] = None):
    """
    Create a default metrics ensemble with sensible defaults
    
    Args:
        models_dir: Directory to save/load models
        
    Returns:
        MetricsEnsembleModel with default models
    """
    ensemble = MetricsEnsembleModel(models_dir=models_dir)
    
    # Common models for each metric
    common_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42)
    ]
    
    # Create ensembles for each metric
    ensemble.set_political_influence_model(
        EnsembleModel(ensemble_type='voting', models=common_models.copy(), models_dir=models_dir)
    )
    
    ensemble.set_rhetoric_intensity_model(
        EnsembleModel(ensemble_type='voting', models=common_models.copy(), models_dir=models_dir)
    )
    
    ensemble.set_information_depth_model(
        EnsembleModel(ensemble_type='voting', models=common_models.copy(), models_dir=models_dir)
    )
    
    return ensemble