"""
Drift Detector module for the Silicon Layer
Detects data drift in features and targets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json

class DriftDetector:
    """
    Drift Detector for monitoring data distributions over time.
    
    Detects:
    - Feature drift
    - Label drift
    - Prediction drift
    """
    
    def __init__(self, reference_data: pd.DataFrame = None, metrics_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the drift detector
        
        Args:
            reference_data: Reference data to compare against
            metrics_dir: Directory to store drift metrics
        """
        self.reference_data = reference_data
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        
        if self.metrics_dir:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set the reference data for drift comparison
        
        Args:
            reference_data: Reference data to compare against
        """
        self.reference_data = reference_data
    
    def calculate_drift_metrics(
        self, 
        current_data: pd.DataFrame, 
        columns: List[str] = None,
        method: str = 'ks',
        threshold: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Calculate drift metrics between reference and current data
        
        Args:
            current_data: Current data to compare against reference
            columns: List of columns to check for drift
            method: Method to use for drift detection ('ks', 'js', 'psi')
            threshold: p-value threshold for significance
            
        Returns:
            Dictionary of drift metrics by column
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        # If columns not specified, use all columns that are in both dataframes
        if columns is None:
            columns = [col for col in self.reference_data.columns if col in current_data.columns]
        
        results = {}
        
        for col in columns:
            if col not in self.reference_data.columns or col not in current_data.columns:
                continue
            
            # Get reference and current values
            ref_values = self.reference_data[col].dropna().values
            curr_values = current_data[col].dropna().values
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Calculate drift metric based on method
            if method == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_detected = p_value < threshold
                
                results[col] = {
                    'method': 'ks',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'threshold': threshold,
                    'drift_detected': drift_detected,
                    'ref_mean': float(ref_values.mean()),
                    'curr_mean': float(curr_values.mean()),
                    'ref_std': float(ref_values.std()),
                    'curr_std': float(curr_values.std())
                }
            
            elif method == 'js':
                # Jensen-Shannon divergence
                # Calculate histograms
                bins = min(20, len(ref_values) // 10, len(curr_values) // 10)
                if bins < 2:
                    bins = 2
                
                hist1, bin_edges = np.histogram(ref_values, bins=bins, density=True)
                hist2, _ = np.histogram(curr_values, bins=bin_edges, density=True)
                
                # Add small epsilon to avoid log(0)
                hist1 = hist1 + 1e-10
                hist2 = hist2 + 1e-10
                
                # Normalize
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # Calculate JS divergence
                m = 0.5 * (hist1 + hist2)
                js_div = 0.5 * (stats.entropy(hist1, m) + stats.entropy(hist2, m))
                
                # JS divergence is between 0 and 1, higher means more drift
                drift_detected = js_div > threshold
                
                results[col] = {
                    'method': 'js',
                    'statistic': float(js_div),
                    'threshold': threshold,
                    'drift_detected': drift_detected,
                    'ref_mean': float(ref_values.mean()),
                    'curr_mean': float(curr_values.mean()),
                    'ref_std': float(ref_values.std()),
                    'curr_std': float(curr_values.std())
                }
            
            elif method == 'psi':
                # Population Stability Index
                bins = min(20, len(ref_values) // 10, len(curr_values) // 10)
                if bins < 2:
                    bins = 2
                
                # Calculate bin edges based on reference data
                bin_edges = np.percentile(ref_values, np.linspace(0, 100, bins + 1))
                bin_edges[0] = -float('inf')
                bin_edges[-1] = float('inf')
                
                # Calculate histograms
                hist1, _ = np.histogram(ref_values, bins=bin_edges)
                hist2, _ = np.histogram(curr_values, bins=bin_edges)
                
                # Convert to percentages
                pct1 = hist1 / len(ref_values)
                pct2 = hist2 / len(curr_values)
                
                # Add small epsilon to avoid division by zero
                pct1 = np.where(pct1 == 0, 1e-10, pct1)
                pct2 = np.where(pct2 == 0, 1e-10, pct2)
                
                # Calculate PSI
                psi = np.sum((pct2 - pct1) * np.log(pct2 / pct1))
                
                # PSI > 0.25 indicates significant drift
                drift_detected = psi > threshold
                
                results[col] = {
                    'method': 'psi',
                    'statistic': float(psi),
                    'threshold': threshold,
                    'drift_detected': drift_detected,
                    'ref_mean': float(ref_values.mean()),
                    'curr_mean': float(curr_values.mean()),
                    'ref_std': float(ref_values.std()),
                    'curr_std': float(curr_values.std())
                }
        
        # Save metrics if directory was provided
        if self.metrics_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.metrics_dir / f"drift_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'method': method,
                    'threshold': threshold,
                    'metrics': results
                }, f, indent=2)
        
        return results
    
    def plot_drift(
        self, 
        current_data: pd.DataFrame,
        columns: List[str] = None,
        max_cols: int = 5,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot distribution drift between reference and current data
        
        Args:
            current_data: Current data to compare against reference
            columns: List of columns to plot
            max_cols: Maximum number of columns to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        # If columns not specified, use all columns that are in both dataframes
        if columns is None:
            columns = [col for col in self.reference_data.columns if col in current_data.columns]
        
        # Limit number of columns to plot
        if len(columns) > max_cols:
            columns = columns[:max_cols]
        
        # Create figure
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(columns):
            if col not in self.reference_data.columns or col not in current_data.columns:
                continue
            
            ax = axes[i]
            
            # Get reference and current values
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Determine if numeric or categorical
            if pd.api.types.is_numeric_dtype(ref_values):
                # Plot numeric column
                sns.kdeplot(ref_values, label='Reference', ax=ax)
                sns.kdeplot(curr_values, label='Current', ax=ax)
                
                # Calculate drift metrics
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                
                ax.set_title(f"{col}\nKS={ks_stat:.3f}, p={p_value:.3f}")
            else:
                # Plot categorical column
                ref_counts = ref_values.value_counts(normalize=True)
                curr_counts = curr_values.value_counts(normalize=True)
                
                # Combine and fill missing categories
                all_cats = pd.Series(index=ref_counts.index.union(curr_counts.index), data=0)
                ref_counts = ref_counts.add(all_cats, fill_value=0)
                curr_counts = curr_counts.add(all_cats, fill_value=0)
                
                # Sort by reference frequency
                cats_sorted = ref_counts.sort_values(ascending=False).index
                
                # Keep only top categories for readability
                top_n = min(10, len(cats_sorted))
                cats_sorted = cats_sorted[:top_n]
                
                # Plot
                width = 0.35
                x = np.arange(len(cats_sorted))
                ax.bar(x - width/2, [ref_counts[c] for c in cats_sorted], width, label='Reference')
                ax.bar(x + width/2, [curr_counts[c] for c in cats_sorted], width, label='Current')
                
                ax.set_xticks(x)
                ax.set_xticklabels([str(c)[:10] for c in cats_sorted], rotation=45, ha='right')
                ax.set_title(f"{col}")
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if directory was provided
        if self.metrics_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.metrics_dir / f"drift_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_drift_summary(self, drift_metrics: Dict[str, Dict]) -> Dict:
        """
        Get summary of drift metrics
        
        Args:
            drift_metrics: Drift metrics from calculate_drift_metrics
            
        Returns:
            Summary dictionary
        """
        num_features = len(drift_metrics)
        num_drifted = sum(1 for col, metric in drift_metrics.items() if metric['drift_detected'])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'num_features': num_features,
            'num_drifted': num_drifted,
            'drift_percentage': num_drifted / num_features if num_features > 0 else 0,
            'drifted_features': [col for col, metric in drift_metrics.items() if metric['drift_detected']]
        }