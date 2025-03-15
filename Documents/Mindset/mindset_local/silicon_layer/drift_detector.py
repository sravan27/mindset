"""
Drift Detector for MINDSET Silicon Layer
Monitors data and concept drift to maintain model reliability.
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
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.silicon_layer.drift_detector')

class DriftDetector:
    """
    Drift detector for monitoring data and concept drift.
    
    This class implements methods to detect data drift (changes in feature distributions)
    and concept drift (changes in the relationship between features and targets).
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        drift_dir: Optional[str] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Baseline data to compare against
            feature_cols: List of feature column names
            target_cols: List of target column names
            drift_dir: Directory to save/load drift detector
            significance_level: Threshold for statistical tests
        """
        self.reference_data = reference_data
        self.feature_cols = feature_cols
        self.target_cols = target_cols or ["political_influence", "rhetoric_intensity", "information_depth"]
        self.drift_dir = drift_dir
        self.significance_level = significance_level
        
        # Feature statistics
        self.feature_stats = {}
        
        # Previous drift results
        self.drift_history = []
        
        if drift_dir:
            os.makedirs(drift_dir, exist_ok=True)
        
        # Initialize reference statistics if reference data is provided
        if reference_data is not None:
            self.compute_reference_statistics()
    
    def compute_reference_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute reference statistics from the baseline data.
        
        Returns:
            Dictionary of feature statistics
        """
        if self.reference_data is None:
            logger.warning("No reference data available. Cannot compute statistics.")
            return {}
        
        # Determine feature columns if not provided
        if self.feature_cols is None:
            # Exclude target columns
            self.feature_cols = [col for col in self.reference_data.columns 
                               if col not in self.target_cols and pd.api.types.is_numeric_dtype(self.reference_data[col])]
            logger.info(f"Inferred {len(self.feature_cols)} feature columns")
        
        # Compute statistics for each feature
        self.feature_stats = {}
        
        for col in self.feature_cols:
            # Get feature values
            values = self.reference_data[col].dropna().values
            
            # Skip if not enough data
            if len(values) < 10:
                logger.warning(f"Not enough data for feature {col}. Skipping.")
                continue
            
            # Compute basic statistics
            self.feature_stats[col] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
                "n_samples": len(values),
                # Keep a histogram for distribution comparison
                "hist": np.histogram(values, bins=20, density=True)
            }
        
        logger.info(f"Computed reference statistics for {len(self.feature_stats)} features")
        return self.feature_stats
    
    def detect_data_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift in new data compared to reference data.
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.feature_stats:
            logger.warning("No reference statistics available. Cannot detect drift.")
            return {"drift_detected": False, "reason": "No reference statistics available"}
        
        drift_results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "drift_detected": False,
            "drifted_features": {},
            "summary": {
                "total_features": len(self.feature_cols),
                "drifted_features_count": 0,
                "drift_percentage": 0
            }
        }
        
        # Check each feature for drift
        for col in self.feature_cols:
            if col not in self.feature_stats:
                continue
                
            # Get new data values
            new_values = new_data[col].dropna().values
            
            # Skip if not enough data
            if len(new_values) < 10:
                logger.warning(f"Not enough new data for feature {col}. Skipping.")
                continue
            
            # Check for distribution drift using KS test
            ks_stat, ks_pvalue = stats.ks_2samp(
                self.reference_data[col].dropna().values,
                new_values
            )
            
            # Check means using t-test
            t_stat, t_pvalue = stats.ttest_ind(
                self.reference_data[col].dropna().values,
                new_values,
                equal_var=False  # Welch's t-test
            )
            
            # Determine if drift is detected
            drift_detected = (ks_pvalue < self.significance_level) or (t_pvalue < self.significance_level)
            
            if drift_detected:
                # Compute statistics for the new data
                new_stats = {
                    "mean": np.mean(new_values),
                    "median": np.median(new_values),
                    "std": np.std(new_values),
                    "min": np.min(new_values),
                    "max": np.max(new_values),
                    "n_samples": len(new_values)
                }
                
                # Calculate differences
                ref_stats = self.feature_stats[col]
                diff_stats = {
                    "mean_diff": new_stats["mean"] - ref_stats["mean"],
                    "mean_diff_pct": (new_stats["mean"] - ref_stats["mean"]) / max(abs(ref_stats["mean"]), 1e-10) * 100,
                    "std_diff": new_stats["std"] - ref_stats["std"],
                    "std_diff_pct": (new_stats["std"] - ref_stats["std"]) / max(abs(ref_stats["std"]), 1e-10) * 100
                }
                
                # Add to results
                drift_results["drifted_features"][col] = {
                    "ks_test": {"statistic": ks_stat, "p_value": ks_pvalue},
                    "t_test": {"statistic": t_stat, "p_value": t_pvalue},
                    "reference_stats": {k: v for k, v in ref_stats.items() if k != "hist"},
                    "new_stats": new_stats,
                    "diff_stats": diff_stats
                }
        
        # Update summary
        drifted_features_count = len(drift_results["drifted_features"])
        drift_results["summary"]["drifted_features_count"] = drifted_features_count
        drift_results["summary"]["drift_percentage"] = (drifted_features_count / len(self.feature_cols)) * 100
        drift_results["drift_detected"] = drifted_features_count > 0
        
        # Add to history
        self.drift_history.append({
            "date": drift_results["date"],
            "drift_detected": drift_results["drift_detected"],
            "drifted_features_count": drifted_features_count
        })
        
        logger.info(f"Data drift detection - Drifted features: {drifted_features_count}/{len(self.feature_cols)} ({drift_results['summary']['drift_percentage']:.2f}%)")
        return drift_results
    
    def detect_concept_drift(
        self, 
        new_data: pd.DataFrame, 
        predictions: pd.DataFrame, 
        actual_values: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect concept drift by comparing model performance on new data.
        
        Args:
            new_data: New feature data
            predictions: Model predictions on new data
            actual_values: Actual values for new data
            
        Returns:
            Dictionary with concept drift detection results
        """
        if not all(target in actual_values.columns for target in self.target_cols):
            logger.warning("Not all target columns available in actual values. Cannot detect concept drift.")
            return {"concept_drift_detected": False, "reason": "Missing target columns"}
        
        if not all(target in predictions.columns for target in self.target_cols):
            logger.warning("Not all target columns available in predictions. Cannot detect concept drift.")
            return {"concept_drift_detected": False, "reason": "Missing prediction columns"}
        
        drift_results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "concept_drift_detected": False,
            "performance_metrics": {},
            "summary": {}
        }
        
        # Check performance for each target
        for target in self.target_cols:
            # Get predictions and actual values
            y_pred = predictions[target].values
            y_true = actual_values[target].values
            
            # Compute performance metrics
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Add to results
            drift_results["performance_metrics"][target] = {
                "mse": mse,
                "r2": r2
            }
            
            # Check if performance has degraded significantly
            # This requires a baseline performance to compare against
            # For now, we'll just use a threshold for R²
            if r2 < 0.5:  # Arbitrary threshold, should be customized
                drift_results["concept_drift_detected"] = True
        
        # Add summary
        drift_results["summary"]["concept_drift_detected"] = drift_results["concept_drift_detected"]
        
        logger.info(f"Concept drift detection - Drift detected: {drift_results['concept_drift_detected']}")
        return drift_results
    
    def generate_drift_report(
        self, 
        data_drift_results: Dict[str, Any],
        concept_drift_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report.
        
        Args:
            data_drift_results: Results from data drift detection
            concept_drift_results: Results from concept drift detection (optional)
            output_path: Path to save the report (optional)
            
        Returns:
            Report dictionary
        """
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_drift": data_drift_results,
            "concept_drift": concept_drift_results,
            "recommendations": []
        }
        
        # Generate recommendations based on drift detection
        if data_drift_results.get("drift_detected", False):
            drift_pct = data_drift_results["summary"]["drift_percentage"]
            
            if drift_pct > 50:
                report["recommendations"].append({
                    "priority": "high",
                    "action": "Retrain model with new data",
                    "reason": f"Significant data drift detected in {drift_pct:.2f}% of features"
                })
            elif drift_pct > 20:
                report["recommendations"].append({
                    "priority": "medium",
                    "action": "Consider retraining with new data",
                    "reason": f"Moderate data drift detected in {drift_pct:.2f}% of features"
                })
            else:
                report["recommendations"].append({
                    "priority": "low",
                    "action": "Monitor data drift",
                    "reason": f"Minor data drift detected in {drift_pct:.2f}% of features"
                })
        
        if concept_drift_results and concept_drift_results.get("concept_drift_detected", False):
            report["recommendations"].append({
                "priority": "high",
                "action": "Retrain model with new data",
                "reason": "Concept drift detected. Model performance has degraded."
            })
        
        # Save report if requested
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save report as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Drift report saved to {output_path}")
        
        return report
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the drift detector to disk.
        
        Args:
            filepath: Path to save the drift detector (optional)
            
        Returns:
            Path where the drift detector was saved
        """
        # Determine filepath
        if filepath is None:
            if self.drift_dir is None:
                raise ValueError("No filepath or drift_dir specified")
            filepath = os.path.join(self.drift_dir, "drift_detector.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save pickled reference statistics and history
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'target_cols': self.target_cols,
                'feature_stats': self.feature_stats,
                'drift_history': self.drift_history,
                'significance_level': self.significance_level
            }, f)
        
        logger.info(f"Drift detector saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'DriftDetector':
        """
        Load a drift detector from disk.
        
        Args:
            filepath: Path to the saved drift detector
            
        Returns:
            Loaded DriftDetector instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance
        instance = cls(
            feature_cols=data['feature_cols'],
            target_cols=data['target_cols'],
            drift_dir=os.path.dirname(filepath),
            significance_level=data['significance_level']
        )
        
        # Load saved data
        instance.feature_stats = data['feature_stats']
        instance.drift_history = data['drift_history']
        
        logger.info(f"Drift detector loaded from {filepath}")
        return instance
    
    def visualize_feature_drift(
        self, 
        new_data: pd.DataFrame, 
        feature: str,
        output_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize drift for a specific feature.
        
        Args:
            new_data: New data to compare against reference
            feature: Feature to visualize
            output_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib Figure or None if error
        """
        if feature not in self.feature_stats:
            logger.warning(f"Feature {feature} not found in reference statistics.")
            return None
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get reference and new data
            ref_data = self.reference_data[feature].dropna()
            new_feature_data = new_data[feature].dropna()
            
            # Create histograms
            ax.hist(ref_data, bins=20, alpha=0.5, label='Reference Data', density=True)
            ax.hist(new_feature_data, bins=20, alpha=0.5, label='New Data', density=True)
            
            # Add statistics
            ref_stats = self.feature_stats[feature]
            ax.axvline(ref_stats['mean'], color='blue', linestyle='dashed', 
                       linewidth=1, label=f'Ref Mean: {ref_stats["mean"]:.2f}')
            
            new_mean = new_feature_data.mean()
            ax.axvline(new_mean, color='orange', linestyle='dashed', 
                       linewidth=1, label=f'New Mean: {new_mean:.2f}')
            
            # Add labels and title
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution Comparison for {feature}')
            ax.legend()
            
            # Add KS test result
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data, new_feature_data)
            drift_detected = ks_pvalue < self.significance_level
            
            drift_text = f"Drift Detected: {drift_detected}\nKS p-value: {ks_pvalue:.4f}"
            ax.text(0.05, 0.95, drift_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"Feature drift visualization saved to {output_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error generating feature drift visualization: {e}")
            return None