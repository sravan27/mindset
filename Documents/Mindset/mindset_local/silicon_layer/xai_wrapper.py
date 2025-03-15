"""
XAI (Explainable AI) Wrapper for MINDSET Silicon Layer
Provides interpretability and explainability for model predictions.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Import explainability libraries
try:
    import shap
    import lime
    import lime.lime_tabular
    EXPLAINERS_AVAILABLE = True
except ImportError:
    EXPLAINERS_AVAILABLE = False
    logging.warning("SHAP and/or LIME not available. Install with: pip install shap lime")

# Local imports
from .ensemble_model import StackedEnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.silicon_layer.xai_wrapper')

class XAIWrapper:
    """
    Explainable AI wrapper for MINDSET models.
    
    This class provides interpretability and explainability for model predictions
    using SHAP and LIME.
    """
    
    def __init__(
        self,
        model: Optional[StackedEnsembleModel] = None,
        feature_cols: Optional[List[str]] = None,
        explainer_dir: Optional[str] = None
    ):
        """
        Initialize the XAI wrapper.
        
        Args:
            model: The model to explain
            feature_cols: List of feature column names
            explainer_dir: Directory to save/load explainers
        """
        self.model = model
        self.feature_cols = feature_cols
        self.explainer_dir = explainer_dir
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if explainer_dir:
            os.makedirs(explainer_dir, exist_ok=True)
    
    def setup_explainers(self, X_background: pd.DataFrame) -> bool:
        """
        Set up SHAP and LIME explainers.
        
        Args:
            X_background: Background data for explainers
            
        Returns:
            True if setup successful, False otherwise
        """
        if not EXPLAINERS_AVAILABLE:
            logger.warning("Cannot set up explainers. SHAP and/or LIME not available.")
            return False
        
        if self.model is None or not self.model.is_fitted:
            logger.warning("Cannot set up explainers. Model not available or not fitted.")
            return False
        
        # Extract feature columns if not explicitly provided
        features = X_background[self.feature_cols] if self.feature_cols else X_background
        
        try:
            # Set up SHAP explainer
            # Use KernelExplainer for flexibility with any model
            logger.info("Setting up SHAP explainer...")
            
            # Use the model's predict method directly instead of a local function
            # This avoids the pickling issue with local functions
            
            # Initialize KernelExplainer with background data
            background_data = shap.sample(features, 100)  # Sample background data for efficiency
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background_data)
            
            # Set up LIME explainer
            logger.info("Setting up LIME explainer...")
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                features.values,
                feature_names=features.columns.tolist(),
                mode="regression"
            )
            
            logger.info("Explainers setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up explainers: {e}")
            return False
    
    def explain_prediction_shap(
        self, 
        X: pd.DataFrame, 
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            X: Features dataframe
            index: Index of the instance to explain
            
        Returns:
            Dictionary with SHAP values and feature contribution details
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized. Call setup_explainers() first.")
            return {}
        
        try:
            # Extract the instance to explain
            instance = X.iloc[[index]]
            features = instance[self.feature_cols] if self.feature_cols else instance
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(features)
            
            # Get the base value (average prediction)
            base_value = self.shap_explainer.expected_value
            
            # Get the actual prediction
            prediction = self.model.predict(features)[0]
            
            # Compile the contribution of each feature
            feature_names = features.columns.tolist()
            contributions = dict(zip(feature_names, shap_values[0]))
            
            # Sort contributions by absolute magnitude
            sorted_contributions = dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            # Create top positive and negative contributions
            pos_contributions = {k: v for k, v in sorted_contributions.items() if v > 0}
            neg_contributions = {k: v for k, v in sorted_contributions.items() if v < 0}
            
            # Compile the explanation
            explanation = {
                "base_value": float(base_value),
                "prediction": float(prediction),
                "contributions": {k: float(v) for k, v in sorted_contributions.items()},
                "top_positive": {k: float(v) for k, v in list(pos_contributions.items())[:5]},
                "top_negative": {k: float(v) for k, v in list(neg_contributions.items())[:5]}
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {}
    
    def explain_prediction_lime(
        self, 
        X: pd.DataFrame, 
        index: int = 0,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction.
        
        Args:
            X: Features dataframe
            index: Index of the instance to explain
            num_features: Number of features to include in the explanation
            
        Returns:
            Dictionary with LIME explanation details
        """
        if self.lime_explainer is None:
            logger.warning("LIME explainer not initialized. Call setup_explainers() first.")
            return {}
        
        try:
            # Extract the instance to explain
            instance = X.iloc[[index]]
            features = instance[self.feature_cols] if self.feature_cols else instance
            
            # Use a class method for prediction to avoid pickling issues
            # Generate LIME explanation using model's predict method directly
            lime_exp = self.lime_explainer.explain_instance(
                features.values[0],
                self.model.predict,
                num_features=num_features
            )
            
            # Extract the explanation details
            feature_importance = lime_exp.as_list()
            
            # Compile the explanation
            explanation = {
                "prediction": float(self.model.predict(features)[0]),
                "feature_importance": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in feature_importance
                ]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {}
    
    def generate_shap_summary_plot(
        self, 
        X: pd.DataFrame,
        plot_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Generate a SHAP summary plot for feature importance.
        
        Args:
            X: Features dataframe
            plot_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib Figure object or None if error
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized. Call setup_explainers() first.")
            return None
        
        try:
            # Extract features
            features = X[self.feature_cols] if self.feature_cols else X
            
            # Generate SHAP values for all instances
            shap_values = self.shap_explainer.shap_values(features)
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            fig = shap.summary_plot(
                shap_values, 
                features,
                show=False
            )
            
            # Save the plot if requested
            if plot_path:
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                logger.info(f"SHAP summary plot saved to {plot_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error generating SHAP summary plot: {e}")
            return None
    
    def save_explainers(self, filepath: Optional[str] = None) -> str:
        """
        Save the explainers to disk.
        
        Args:
            filepath: Path to save the explainers (optional)
            
        Returns:
            Path where the explainers were saved
        """
        if self.shap_explainer is None and self.lime_explainer is None:
            logger.warning("No explainers to save.")
            return ""
        
        # Determine filepath
        if filepath is None:
            if self.explainer_dir is None:
                raise ValueError("No filepath or explainer_dir specified")
            filepath = os.path.join(self.explainer_dir, "explainers.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save explainers to disk
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'shap_explainer': self.shap_explainer,
                'lime_explainer': self.lime_explainer
            }, f)
        
        logger.info(f"Explainers saved to {filepath}")
        return filepath
    
    @classmethod
    def load(
        cls, 
        filepath: str,
        model: StackedEnsembleModel
    ) -> 'XAIWrapper':
        """
        Load explainers from disk.
        
        Args:
            filepath: Path to the saved explainers
            model: The model to use with the explainers
            
        Returns:
            Loaded XAIWrapper instance
        """
        with open(filepath, 'rb') as f:
            explainer_data = pickle.load(f)
        
        # Create new instance
        instance = cls(
            model=model,
            feature_cols=explainer_data['feature_cols'],
            explainer_dir=os.path.dirname(filepath)
        )
        
        # Load explainer components
        instance.shap_explainer = explainer_data['shap_explainer']
        instance.lime_explainer = explainer_data['lime_explainer']
        
        logger.info(f"Explainers loaded from {filepath}")
        return instance
    
    def get_feature_importance(self, X: pd.DataFrame, method: str = 'shap') -> Dict[str, float]:
        """
        Get overall feature importance.
        
        Args:
            X: Features dataframe
            method: Method to use for feature importance ('shap' or 'permutation')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if method == 'shap':
            return self._get_shap_feature_importance(X)
        else:
            # Fallback to model's feature importance if SHAP not available
            if hasattr(self.model, 'get_feature_importance'):
                return self.model.get_feature_importance()
            else:
                logger.warning("Feature importance not available for this model.")
                return {}
    
    def _get_shap_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance using SHAP values."""
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized. Call setup_explainers() first.")
            return {}
        
        try:
            # Extract features
            features = X[self.feature_cols] if self.feature_cols else X
            
            # Sample data for efficiency if needed
            if len(features) > 100:
                features = features.sample(100, random_state=42)
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(features)
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create dictionary of feature importance
            importance_dict = dict(zip(features.columns, mean_abs_shap))
            
            # Sort by importance (descending)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating SHAP feature importance: {e}")
            return {}