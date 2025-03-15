"""
Silicon Layer Integrator for MINDSET
Orchestrates model training, evaluation, and deployment.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Local imports
from .ensemble_model import StackedEnsembleModel, MetricEnsembleTrainer
from .xai_wrapper import XAIWrapper
from .drift_detector import DriftDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.silicon_layer')

class SiliconLayer:
    """
    Silicon Layer for MINDSET.
    
    This class orchestrates the ML operations for MINDSET, including model training,
    evaluation, explainability, and drift detection.
    """
    
    def __init__(
        self,
        base_dir: str,
        data_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        explainer_dir: Optional[str] = None,
        drift_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize the Silicon Layer.
        
        Args:
            base_dir: Base directory for MINDSET
            data_dir: Directory for data (default: {base_dir}/data)
            model_dir: Directory for models (default: {base_dir}/models)
            explainer_dir: Directory for explainers (default: {base_dir}/explainers)
            drift_dir: Directory for drift detection (default: {base_dir}/drift)
            random_state: Random seed for reproducibility
        """
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data"
        self.model_dir = Path(model_dir) if model_dir else self.base_dir / "models"
        self.explainer_dir = Path(explainer_dir) if explainer_dir else self.base_dir / "explainers"
        self.drift_dir = Path(drift_dir) if drift_dir else self.base_dir / "drift"
        self.random_state = random_state
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.explainer_dir.mkdir(exist_ok=True, parents=True)
        self.drift_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.ensemble_trainer = None
        self.xai_wrapper = None
        self.drift_detector = None
        
        # Model evaluation metrics
        self.evaluation_metrics = {}
        
        logger.info("Silicon Layer initialized")
    
    def train_models(
        self, 
        train_data: pd.DataFrame, 
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        create_explainers: bool = True,
        setup_drift_detector: bool = True
    ) -> Dict[str, Any]:
        """
        Train models for all metrics.
        
        Args:
            train_data: Training data with features and targets
            feature_cols: List of feature column names (optional)
            test_size: Fraction of data to use for testing
            create_explainers: Whether to create explainers for the models
            setup_drift_detector: Whether to set up drift detection
            
        Returns:
            Dictionary with training and evaluation results
        """
        logger.info("Starting model training")
        
        # Split data into train and test sets
        train_df, test_df = train_test_split(
            train_data, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        logger.info(f"Training set: {len(train_df)} samples, Test set: {len(test_df)} samples")
        
        # Initialize ensemble trainer
        self.ensemble_trainer = MetricEnsembleTrainer(
            base_dir=str(self.model_dir),
            feature_cols=feature_cols,
            random_state=self.random_state
        )
        
        # Train models
        model_paths = self.ensemble_trainer.train_models(train_df)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(test_df)
        
        # Set up explainers if requested
        if create_explainers:
            # Use the first trained model for explainability
            first_model_key = list(self.ensemble_trainer.models.keys())[0]
            first_model = self.ensemble_trainer.models[first_model_key]
            
            # Initialize XAI wrapper
            self.xai_wrapper = XAIWrapper(
                model=first_model,
                feature_cols=self.ensemble_trainer.feature_cols,
                explainer_dir=str(self.explainer_dir)
            )
            
            # Set up explainers
            explainer_setup = self.xai_wrapper.setup_explainers(test_df)
            
            if explainer_setup:
                # Save explainers
                explainer_path = self.xai_wrapper.save_explainers()
                logger.info(f"Explainers saved to {explainer_path}")
                
                # Generate feature importance visualization
                if len(test_df) > 10:
                    sample_data = test_df.sample(10, random_state=self.random_state)
                else:
                    sample_data = test_df
                
                feature_importance = self.xai_wrapper.get_feature_importance(sample_data)
                logger.info(f"Top 5 important features: {list(feature_importance.items())[:5]}")
        
        # Set up drift detector if requested
        if setup_drift_detector:
            # Initialize drift detector
            self.drift_detector = DriftDetector(
                reference_data=test_df,
                feature_cols=self.ensemble_trainer.feature_cols,
                drift_dir=str(self.drift_dir),
                significance_level=0.05
            )
            
            # Compute reference statistics
            self.drift_detector.compute_reference_statistics()
            
            # Save drift detector
            drift_detector_path = self.drift_detector.save()
            logger.info(f"Drift detector saved to {drift_detector_path}")
        
        # Export model metadata
        metadata_path = self.ensemble_trainer.export_model_metadata()
        
        # Compile results
        results = {
            "model_paths": model_paths,
            "evaluation": evaluation_results,
            "metadata_path": metadata_path
        }
        
        logger.info("Model training completed successfully")
        return results
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data.
        
        Args:
            test_data: Test data with features and ground truth
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.ensemble_trainer is None or not any(model for model in self.ensemble_trainer.models.values()):
            logger.warning("No trained models available. Call train_models() first.")
            return {}
        
        logger.info("Evaluating models on test data")
        
        # Make predictions
        predictions = self.ensemble_trainer.predict(test_data)
        
        # Calculate metrics for each target
        eval_metrics = {}
        
        for target in predictions.columns:
            if target == 'information_depth_category':
                continue  # Skip categorical output
                
            if target not in test_data.columns:
                logger.warning(f"Target {target} not found in test data. Skipping evaluation.")
                continue
            
            # Extract predictions and ground truth
            y_pred = predictions[target].values
            y_true = test_data[target].values
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Store metrics
            eval_metrics[target] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            logger.info(f"Metrics for {target}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
        
        # Store evaluation metrics
        self.evaluation_metrics = eval_metrics
        
        return eval_metrics
    
    def load_models(self) -> Dict[str, str]:
        """
        Load trained models.
        
        Returns:
            Dictionary mapping metric names to model paths
        """
        # Initialize ensemble trainer if not already done
        if self.ensemble_trainer is None:
            self.ensemble_trainer = MetricEnsembleTrainer(
                base_dir=str(self.model_dir),
                random_state=self.random_state
            )
        
        # Load models
        models = self.ensemble_trainer.load_models()
        
        # Check if models were loaded successfully
        loaded_models = {
            name: str(self.model_dir / f"{name}_model.pkl")
            for name, model in models.items()
            if model is not None and hasattr(model, 'is_fitted') and model.is_fitted
        }
        
        if not loaded_models:
            logger.warning("No models loaded. Train models first or check model directory.")
            return {}
        
        logger.info(f"Loaded {len(loaded_models)} models")
        return loaded_models
    
    def load_explainers(self) -> bool:
        """
        Load explainers for model interpretability.
        
        Returns:
            True if explainers loaded successfully, False otherwise
        """
        # Check if models are loaded
        if self.ensemble_trainer is None or not any(model for model in self.ensemble_trainer.models.values()):
            logger.warning("No models loaded. Load models first.")
            return False
        
        # Get the first loaded model for explainability
        first_model_key = None
        for key, model in self.ensemble_trainer.models.items():
            if model is not None and hasattr(model, 'is_fitted') and model.is_fitted:
                first_model_key = key
                break
        
        if first_model_key is None:
            logger.warning("No valid models found. Cannot load explainers.")
            return False
        
        # Path to explainers
        explainer_path = self.explainer_dir / "explainers.pkl"
        
        if not explainer_path.exists():
            logger.warning(f"Explainers not found at {explainer_path}")
            return False
        
        try:
            # Load explainers
            self.xai_wrapper = XAIWrapper.load(
                str(explainer_path),
                model=self.ensemble_trainer.models[first_model_key]
            )
            
            logger.info("Explainers loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading explainers: {e}")
            return False
    
    def load_drift_detector(self) -> bool:
        """
        Load drift detector.
        
        Returns:
            True if drift detector loaded successfully, False otherwise
        """
        # Path to drift detector
        drift_detector_path = self.drift_dir / "drift_detector.pkl"
        
        if not drift_detector_path.exists():
            logger.warning(f"Drift detector not found at {drift_detector_path}")
            return False
        
        try:
            # Load drift detector
            self.drift_detector = DriftDetector.load(str(drift_detector_path))
            
            logger.info("Drift detector loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading drift detector: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with trained models.
        
        Args:
            data: Feature data
            
        Returns:
            DataFrame with predictions
        """
        if self.ensemble_trainer is None:
            # Try to load models
            self.load_models()
            
            if self.ensemble_trainer is None or not any(model for model in self.ensemble_trainer.models.values()):
                logger.error("No models loaded. Cannot make predictions.")
                return pd.DataFrame()
        
        # Make predictions
        try:
            predictions = self.ensemble_trainer.predict(data)
            
            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame()
    
    def explain_prediction(
        self, 
        data: pd.DataFrame, 
        index: int = 0,
        explanation_type: str = 'shap'
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            data: Feature data
            index: Index of the sample to explain
            explanation_type: Type of explanation ('shap' or 'lime')
            
        Returns:
            Dictionary with explanation details
        """
        if self.xai_wrapper is None:
            # Try to load explainers
            self.load_explainers()
            
            if self.xai_wrapper is None:
                logger.error("No explainers loaded. Cannot explain prediction.")
                return {}
        
        # Generate explanation
        try:
            if explanation_type.lower() == 'shap':
                explanation = self.xai_wrapper.explain_prediction_shap(data, index)
            elif explanation_type.lower() == 'lime':
                explanation = self.xai_wrapper.explain_prediction_lime(data, index)
            else:
                logger.warning(f"Unknown explanation type: {explanation_type}. Using SHAP.")
                explanation = self.xai_wrapper.explain_prediction_shap(data, index)
            
            logger.info(f"Generated {explanation_type} explanation for sample {index}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {}
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift in new data.
        
        Args:
            new_data: New feature data
            
        Returns:
            Dictionary with drift detection results
        """
        if self.drift_detector is None:
            # Try to load drift detector
            self.load_drift_detector()
            
            if self.drift_detector is None:
                logger.error("No drift detector loaded. Cannot detect drift.")
                return {"drift_detected": False, "reason": "No drift detector loaded"}
        
        # Detect drift
        try:
            drift_results = self.drift_detector.detect_data_drift(new_data)
            
            # Generate drift report
            drift_report_path = self.drift_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            drift_report = self.drift_detector.generate_drift_report(
                drift_results,
                output_path=str(drift_report_path)
            )
            
            logger.info(f"Drift detection completed. Report saved to {drift_report_path}")
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {"drift_detected": False, "reason": str(e)}
    
    def monitor_performance(
        self, 
        new_data: pd.DataFrame, 
        actual_values: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Monitor model performance and detect concept drift.
        
        Args:
            new_data: New feature data
            actual_values: Actual values for new data
            
        Returns:
            Dictionary with monitoring results
        """
        if self.ensemble_trainer is None or not any(model for model in self.ensemble_trainer.models.values()):
            logger.warning("No models loaded. Cannot monitor performance.")
            return {"status": "error", "reason": "No models loaded"}
        
        if self.drift_detector is None:
            # Try to load drift detector
            self.load_drift_detector()
            
            if self.drift_detector is None:
                logger.warning("No drift detector loaded. Will only report performance metrics.")
        
        # Make predictions
        predictions = self.predict(new_data)
        
        if len(predictions) == 0:
            logger.error("Failed to make predictions. Cannot monitor performance.")
            return {"status": "error", "reason": "Failed to make predictions"}
        
        # Calculate performance metrics
        performance_metrics = {}
        
        for target in predictions.columns:
            if target == 'information_depth_category':
                continue  # Skip categorical output
                
            if target not in actual_values.columns:
                logger.warning(f"Target {target} not found in actual values. Skipping.")
                continue
            
            # Extract predictions and actual values
            y_pred = predictions[target].values
            y_true = actual_values[target].values
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Store metrics
            performance_metrics[target] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            logger.info(f"Current metrics for {target}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
        
        # Detect concept drift if drift detector is available
        concept_drift_results = None
        
        if self.drift_detector is not None:
            try:
                concept_drift_results = self.drift_detector.detect_concept_drift(
                    new_data, predictions, actual_values
                )
            except Exception as e:
                logger.error(f"Error detecting concept drift: {e}")
        
        # Compile monitoring results
        monitoring_results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance_metrics": performance_metrics,
            "concept_drift": concept_drift_results
        }
        
        # Save monitoring results
        monitoring_path = self.drift_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(monitoring_path, 'w') as f:
                json.dump(monitoring_results, f, indent=2)
            
            logger.info(f"Monitoring results saved to {monitoring_path}")
        except Exception as e:
            logger.error(f"Error saving monitoring results: {e}")
        
        return monitoring_results
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Report dictionary
        """
        if not self.evaluation_metrics:
            logger.warning("No evaluation metrics available. Run evaluate_models() first.")
            return {}
        
        # Compile report
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_metrics": self.evaluation_metrics,
            "feature_importance": {}
        }
        
        # Add feature importance if available
        if self.xai_wrapper is not None:
            try:
                # Get feature importance
                # We need some data for this, so we'll use dummy data
                dummy_data = pd.DataFrame(
                    np.random.random((10, len(self.xai_wrapper.feature_cols))),
                    columns=self.xai_wrapper.feature_cols
                )
                
                report["feature_importance"] = self.xai_wrapper.get_feature_importance(dummy_data)
            except Exception as e:
                logger.error(f"Error getting feature importance: {e}")
        
        # Save report if requested
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save report as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report saved to {output_path}")
        
        return report
    
    def export_models_for_api(self, export_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export models for use in the API.
        
        Args:
            export_dir: Directory to export models (optional)
            
        Returns:
            Dictionary mapping metric names to exported model paths
        """
        if self.ensemble_trainer is None:
            # Try to load models
            self.load_models()
            
            if self.ensemble_trainer is None or not any(model for model in self.ensemble_trainer.models.values()):
                logger.error("No models loaded. Cannot export models.")
                return {}
        
        # Determine export directory
        if export_dir is None:
            export_dir = self.base_dir / "data" / "gold" / "models"
        else:
            export_dir = Path(export_dir)
        
        # Create export directory
        export_dir.mkdir(exist_ok=True, parents=True)
        
        # Export models
        exported_paths = {}
        
        for name, model in self.ensemble_trainer.models.items():
            if model is None or not hasattr(model, 'is_fitted') or not model.is_fitted:
                logger.warning(f"Model {name} is not fitted. Skipping export.")
                continue
            
            # Export path
            export_path = export_dir / f"{name}_model.pkl"
            
            try:
                # Save model
                model.save(str(export_path))
                
                # Add to exported paths
                exported_paths[name] = str(export_path)
                
                logger.info(f"Model {name} exported to {export_path}")
            except Exception as e:
                logger.error(f"Error exporting model {name}: {e}")
        
        # Export model metadata
        metadata_path = export_dir / "model_metadata.json"
        
        try:
            self.ensemble_trainer.export_model_metadata()
            logger.info(f"Model metadata exported to {metadata_path}")
        except Exception as e:
            logger.error(f"Error exporting model metadata: {e}")
        
        return exported_paths