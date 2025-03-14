"""
XAI Wrapper module for the Silicon Layer
Provides explainability for ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")

class XAIWrapper:
    """
    Explainable AI wrapper for Silicon Layer models
    
    Provides:
    - SHAP explanations
    - Feature importance
    - Custom explanations for news metrics
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str] = None,
        explainer_type: str = 'tree',
        explainer_args: Dict = None,
        explanations_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the XAI wrapper
        
        Args:
            model: Model to explain
            feature_names: List of feature names
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'deep', etc.)
            explainer_args: Additional arguments for the explainer
            explanations_dir: Directory to save explanations
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer_args = explainer_args or {}
        self.explanations_dir = Path(explanations_dir) if explanations_dir else None
        
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        if self.explanations_dir:
            self.explanations_dir.mkdir(parents=True, exist_ok=True)
    
    def create_explainer(self, background_data: np.ndarray = None):
        """
        Create the SHAP explainer
        
        Args:
            background_data: Background data for the explainer
            
        Returns:
            self
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return self
        
        self.background_data = background_data
        
        # Create the explainer based on the type
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model, **self.explainer_args)
        
        elif self.explainer_type == 'kernel':
            if background_data is None:
                raise ValueError("background_data must be provided for kernel explainer")
            
            self.explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, 'predict') else self.model,
                background_data,
                **self.explainer_args
            )
        
        elif self.explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data, **self.explainer_args)
        
        elif self.explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, background_data, **self.explainer_args)
        
        else:
            raise ValueError(f"Unsupported explainer type: {self.explainer_type}")
        
        return self
    
    def explain(self, X: np.ndarray, nsamples: int = 100):
        """
        Generate SHAP values for the given data
        
        Args:
            X: Data to explain
            nsamples: Number of samples for kernel explainer
            
        Returns:
            self
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return self
        
        if self.explainer is None:
            self.create_explainer(X[:min(100, len(X))])
        
        if self.explainer_type == 'kernel':
            self.shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        return self
    
    def plot_summary(self, X: np.ndarray = None, max_display: int = 20, title: str = None):
        """
        Generate a summary plot of SHAP values
        
        Args:
            X: Data to explain (if not already explained)
            max_display: Maximum number of features to display
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return None
        
        # Generate SHAP values if not already done
        if self.shap_values is None and X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first or provide X.")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot summary
        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        feature_names = self.feature_names if self.feature_names else None
        
        shap.summary_plot(
            shap_values,
            features=X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        
        # Save plot if directory was provided
        if self.explanations_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.explanations_dir / f"shap_summary_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_dependence(self, feature_idx: int, interaction_idx: str = "auto", X: np.ndarray = None):
        """
        Generate a dependence plot for a specific feature
        
        Args:
            feature_idx: Index or name of the feature
            interaction_idx: Index or name of the interaction feature
            X: Data to explain (if not already explained)
            
        Returns:
            matplotlib figure
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return None
        
        # Generate SHAP values if not already done
        if self.shap_values is None and X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first or provide X.")
        
        # If feature_idx is a string, convert to index
        if isinstance(feature_idx, str) and self.feature_names:
            try:
                feature_idx = self.feature_names.index(feature_idx)
            except ValueError:
                raise ValueError(f"Feature '{feature_idx}' not found in feature_names")
        
        # If interaction_idx is a string (not "auto") and not a feature name, convert to index
        if interaction_idx != "auto" and isinstance(interaction_idx, str) and self.feature_names:
            try:
                interaction_idx = self.feature_names.index(interaction_idx)
            except ValueError:
                raise ValueError(f"Feature '{interaction_idx}' not found in feature_names")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot dependence
        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        feature_names = self.feature_names if self.feature_names else None
        
        shap.dependence_plot(
            feature_idx,
            shap_values,
            features=X,
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        
        plt.tight_layout()
        
        # Save plot if directory was provided
        if self.explanations_dir:
            feature_name = feature_idx
            if self.feature_names and isinstance(feature_idx, int):
                feature_name = self.feature_names[feature_idx]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.explanations_dir / f"shap_dependence_{feature_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_force(self, idx: int, X: np.ndarray = None):
        """
        Generate a force plot for a specific instance
        
        Args:
            idx: Index of the instance
            X: Data to explain (if not already explained)
            
        Returns:
            matplotlib figure
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return None
        
        # Generate SHAP values if not already done
        if self.shap_values is None and X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first or provide X.")
        
        # Create figure
        plt.figure(figsize=(15, 3))
        
        # Plot force plot
        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        # Temporary workaround for MatplotlibDeprecationWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot = shap.force_plot(
                self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                shap_values[idx],
                features=X[idx] if X is not None else None,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        
        # Save plot if directory was provided
        if self.explanations_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.explanations_dir / f"shap_force_{idx}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_feature_importance(self, X: np.ndarray = None, aggfunc: str = 'mean_abs'):
        """
        Get feature importance based on SHAP values
        
        Args:
            X: Data to explain (if not already explained)
            aggfunc: Aggregation function for SHAP values ('mean_abs', 'mean', 'sum_abs', 'sum')
            
        Returns:
            DataFrame with feature importance
        """
        if not SHAP_AVAILABLE:
            # Fallback to model's feature_importances_ if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                features = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
                return pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
            else:
                warnings.warn("SHAP is not installed and model doesn't have feature_importances_. Install SHAP with 'pip install shap'.")
                return None
        
        # Generate SHAP values if not already done
        if self.shap_values is None and X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first or provide X.")
        
        # Extract SHAP values
        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        # Calculate feature importance
        if aggfunc == 'mean_abs':
            importances = np.mean(np.abs(shap_values), axis=0)
        elif aggfunc == 'mean':
            importances = np.mean(shap_values, axis=0)
        elif aggfunc == 'sum_abs':
            importances = np.sum(np.abs(shap_values), axis=0)
        elif aggfunc == 'sum':
            importances = np.sum(shap_values, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation function: {aggfunc}")
        
        # Create DataFrame
        features = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
        importance_df = pd.DataFrame({'feature': features, 'importance': importances})
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, X: np.ndarray, idx: int, top_n: int = 5):
        """
        Generate a human-readable explanation for a specific prediction
        
        Args:
            X: Data to explain
            idx: Index of the instance to explain
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanation
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP is not installed. Install with 'pip install shap' for model explanations.")
            return {"explanation": "Model explanations not available. Install SHAP for detailed explanations."}
        
        # Generate SHAP values if not already done
        if self.shap_values is None:
            self.explain(X)
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(X[idx].reshape(1, -1))[0]
        else:
            prediction = self.model(X[idx].reshape(1, -1))[0]
        
        # Extract SHAP values for the instance
        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        instance_shap = shap_values[idx]
        
        # Get base value
        base_value = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) == 1:
            base_value = base_value[0]
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(instance_shap))]
        
        # Create explanation
        feature_contributions = []
        for i, (value, name) in enumerate(zip(instance_shap, feature_names)):
            feature_contributions.append({
                'feature': name,
                'contribution': float(value),
                'value': float(X[idx, i]) if X is not None else None
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Get top contributing features
        top_features = feature_contributions[:top_n]
        
        # Generate explanation text
        positive_features = [f for f in top_features if f['contribution'] > 0]
        negative_features = [f for f in top_features if f['contribution'] < 0]
        
        explanation_text = []
        if positive_features:
            positive_text = ", ".join([f"{f['feature']}" for f in positive_features])
            explanation_text.append(f"The main factors increasing the prediction were: {positive_text}.")
        
        if negative_features:
            negative_text = ", ".join([f"{f['feature']}" for f in negative_features])
            explanation_text.append(f"The main factors decreasing the prediction were: {negative_text}.")
        
        # Create explanation object
        explanation = {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'explanation_text': explanation_text,
            'top_features': top_features,
            'all_features': feature_contributions
        }
        
        return explanation


class MetricsExplainer:
    """
    Combined explainer for all news metrics
    
    Explains:
    - Political Influence Level
    - Rhetoric Intensity Scale
    - Information Depth Score
    """
    
    def __init__(
        self,
        models_dict: Dict,
        feature_names: List[str] = None,
        explainer_type: str = 'tree',
        explanations_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the metrics explainer
        
        Args:
            models_dict: Dictionary of models for each metric
            feature_names: List of feature names
            explainer_type: Type of SHAP explainer
            explanations_dir: Directory to save explanations
        """
        self.models_dict = models_dict
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explanations_dir = Path(explanations_dir) if explanations_dir else None
        
        # Create explainers for each metric
        self.explainers = {}
        
        for metric_name, model in self.models_dict.items():
            metric_dir = None
            if self.explanations_dir:
                metric_dir = self.explanations_dir / metric_name
                metric_dir.mkdir(parents=True, exist_ok=True)
            
            self.explainers[metric_name] = XAIWrapper(
                model=model,
                feature_names=feature_names,
                explainer_type=explainer_type,
                explanations_dir=metric_dir
            )
    
    def create_explainers(self, background_data: np.ndarray = None):
        """
        Create all explainers
        
        Args:
            background_data: Background data for the explainers
            
        Returns:
            self
        """
        for metric_name, explainer in self.explainers.items():
            explainer.create_explainer(background_data)
        
        return self
    
    def explain_all(self, X: np.ndarray, nsamples: int = 100):
        """
        Generate explanations for all metrics
        
        Args:
            X: Data to explain
            nsamples: Number of samples for kernel explainer
            
        Returns:
            self
        """
        for metric_name, explainer in self.explainers.items():
            explainer.explain(X, nsamples=nsamples)
        
        return self
    
    def plot_all_summaries(self, X: np.ndarray = None, max_display: int = 20):
        """
        Generate summary plots for all metrics
        
        Args:
            X: Data to explain (if not already explained)
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        for metric_name, explainer in self.explainers.items():
            figures[metric_name] = explainer.plot_summary(
                X=X,
                max_display=max_display,
                title=f"{metric_name.replace('_', ' ').title()} - Feature Importance"
            )
        
        return figures
    
    def get_all_feature_importance(self, X: np.ndarray = None, aggfunc: str = 'mean_abs'):
        """
        Get feature importance for all metrics
        
        Args:
            X: Data to explain (if not already explained)
            aggfunc: Aggregation function for SHAP values
            
        Returns:
            Dictionary of DataFrames with feature importance
        """
        importance_dfs = {}
        
        for metric_name, explainer in self.explainers.items():
            importance_dfs[metric_name] = explainer.get_feature_importance(X=X, aggfunc=aggfunc)
        
        return importance_dfs
    
    def explain_article(self, X: np.ndarray, idx: int, top_n: int = 5):
        """
        Generate explanations for a specific article across all metrics
        
        Args:
            X: Data to explain
            idx: Index of the article
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanations for each metric
        """
        explanations = {}
        
        for metric_name, explainer in self.explainers.items():
            explanations[metric_name] = explainer.explain_prediction(X=X, idx=idx, top_n=top_n)
        
        return explanations