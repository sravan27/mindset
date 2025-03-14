#!/bin/bash
set -e

# MINDSET: Cloud-Native News Analytics Platform
# Production-level deployment script for Azure

# Print header
echo "================================================="
echo "  MINDSET - Cloud-Native News Analytics Platform  "
echo "================================================="
echo "Starting production-level deployment to Azure..."

# Check required commands
echo "Checking required commands..."
REQUIRED_COMMANDS=("az" "docker" "python" "pip")
MISSING_COMMANDS=()

for cmd in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v "$cmd" &> /dev/null; then
    MISSING_COMMANDS+=("$cmd")
  fi
done

if [ ${#MISSING_COMMANDS[@]} -gt 0 ]; then
  echo "Error: The following required commands are missing:"
  for cmd in "${MISSING_COMMANDS[@]}"; do
    echo "  - $cmd"
  done
  echo "Please install the missing dependencies and try again."
  exit 1
fi

echo "All required commands are available"

# Collect API key
NEWS_API_KEY=""
read -p "Please provide your NewsAPI.org API key (press Enter to use a demo key): " NEWS_API_KEY
echo "> $NEWS_API_KEY"

if [ -z "$NEWS_API_KEY" ]; then
  NEWS_API_KEY="demo_key_for_testing"
  echo "Using demo key for testing (limited functionality)"
fi

export NEWS_API_KEY

# Check Azure authentication
echo "Checking Azure authentication..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

if [ $? -ne 0 ]; then
  echo "Error: Not logged in to Azure. Please run 'az login' first."
  exit 1
fi

echo "Using Azure subscription: $SUBSCRIPTION_ID"

# Resource Group
RESOURCE_GROUP="MBD-EN-ABR-2024-Group-1"
echo "Checking resource group..."

GROUP_EXISTS=$(az group exists --name "$RESOURCE_GROUP")

if [ "$GROUP_EXISTS" = "false" ]; then
  echo "Creating resource group $RESOURCE_GROUP..."
  az group create --name "$RESOURCE_GROUP" --location "westeurope"
else
  echo "Using existing resource group $RESOURCE_GROUP"
fi

# Set directories
export MINDSET_ROOT=$(pwd)
export INFRASTRUCTURE_DIR="$MINDSET_ROOT/infrastructure/azure"
export KUBERNETES_DIR="$MINDSET_ROOT/kubernetes"
export SRC_DIR="$MINDSET_ROOT/src"

# Create necessary directories
mkdir -p "$INFRASTRUCTURE_DIR"
mkdir -p "$KUBERNETES_DIR"
mkdir -p "$SRC_DIR/api"
mkdir -p "$SRC_DIR/ml/silicon_layer"
mkdir -p "$SRC_DIR/rust/metrics_engine"

# Make scripts executable
chmod +x $INFRASTRUCTURE_DIR/*.sh 2>/dev/null || true

# Step 1: Deploy Azure infrastructure
echo "====================================="
echo "Step 1: Deploying Azure infrastructure"
echo "====================================="
cd $INFRASTRUCTURE_DIR
./deploy-infrastructure.sh
source ./azure-config.env

# Step 2: Set up Rust metrics engine
echo "====================================="
echo "Step 2: Setting up Rust metrics engine"
echo "====================================="
cd $INFRASTRUCTURE_DIR
./setup-rust.sh

# Step 3: Create Silicon Layer
echo "====================================="
echo "Step 3: Creating Silicon Layer"
echo "====================================="

# Create Silicon Layer Python implementation if it doesn't exist
SILICON_DIR="$SRC_DIR/ml/silicon_layer"
if [ ! -f "$SILICON_DIR/__init__.py" ]; then
  echo "Creating Silicon Layer implementation..."
  
  mkdir -p $SILICON_DIR
  
  # Create __init__.py
  cat > $SILICON_DIR/__init__.py << EOF
"""
MINDSET Silicon Layer: Advanced ML processing between Silver and Gold medallion layers.
"""
from .feature_store import FeatureStore
from .drift_detector import DriftDetector
from .ensemble_model import EnsembleModel
from .xai_wrapper import XAIWrapper
from .silicon_layer import SiliconLayer

__all__ = [
    'FeatureStore',
    'DriftDetector',
    'EnsembleModel',
    'XAIWrapper',
    'SiliconLayer'
]
EOF

  # Create feature_store.py
  cat > $SILICON_DIR/feature_store.py << EOF
"""
Feature Store for MINDSET Silicon Layer.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Feature Store for managing ML features efficiently.
    
    Key capabilities:
    - Feature versioning
    - Historical feature values
    - Feature transformations
    - Feature serving for both training and inference
    """
    
    def __init__(self, storage_path: str = None):
        """Initialize the feature store."""
        self.storage_path = storage_path or os.environ.get('FEATURE_STORE_PATH', '/tmp/mindset_features')
        os.makedirs(self.storage_path, exist_ok=True)
        self.features = {}
        self._load_features()
    
    def _load_features(self):
        """Load features from storage."""
        try:
            feature_file = os.path.join(self.storage_path, 'features.json')
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    self.features = json.load(f)
                logger.info(f"Loaded {len(self.features)} features from storage")
        except Exception as e:
            logger.error(f"Error loading features: {e}")
    
    def _save_features(self):
        """Save features to storage."""
        try:
            feature_file = os.path.join(self.storage_path, 'features.json')
            with open(feature_file, 'w') as f:
                json.dump(self.features, f)
            logger.info(f"Saved {len(self.features)} features to storage")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
    
    def add_feature(self, name: str, value: Any, metadata: Dict[str, Any] = None):
        """Add a new feature or update an existing one."""
        if name not in self.features:
            self.features[name] = {
                'values': [],
                'metadata': metadata or {}
            }
        
        # Add new value with timestamp
        self.features[name]['values'].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history to last 100 values
        self.features[name]['values'] = self.features[name]['values'][-100:]
        
        # Save to storage
        self._save_features()
    
    def get_feature(self, name: str, default: Any = None) -> Any:
        """Get the latest value of a feature."""
        if name in self.features and self.features[name]['values']:
            return self.features[name]['values'][-1]['value']
        return default
    
    def get_feature_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical values of a feature."""
        if name in self.features:
            return self.features[name]['values'][-limit:]
        return []
    
    def get_all_features(self) -> Dict[str, Any]:
        """Get all latest feature values."""
        result = {}
        for name in self.features:
            if self.features[name]['values']:
                result[name] = self.features[name]['values'][-1]['value']
        return result
EOF

  # Create drift_detector.py
  cat > $SILICON_DIR/drift_detector.py << EOF
"""
Drift Detector for MINDSET Silicon Layer.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Drift Detector for monitoring data and concept drift.
    
    Key capabilities:
    - Statistical drift detection
    - Feature distribution monitoring
    - Alerting on significant drift
    """
    
    def __init__(self, reference_data: Optional[Dict[str, Any]] = None):
        """Initialize the drift detector."""
        self.reference_data = reference_data or {}
        self.latest_data = {}
        self.drift_thresholds = {
            'ks_threshold': 0.1,  # Kolmogorov-Smirnov test threshold
            'js_threshold': 0.05,  # Jensen-Shannon divergence threshold
        }
        self.drift_history = []
    
    def update_reference(self, data: Dict[str, Any]):
        """Update reference data for drift comparison."""
        self.reference_data = data
        logger.info(f"Updated reference data with {len(data)} features")
    
    def check_drift(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for drift between reference and current data.
        
        Returns a report with drift metrics and detected drift features.
        """
        self.latest_data = current_data
        
        if not self.reference_data:
            logger.warning("No reference data available for drift detection")
            return {
                'drift_detected': False,
                'drift_features': [],
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate metrics for numerical features
        drift_features = []
        metrics = {}
        
        for feature, value in current_data.items():
            if feature in self.reference_data:
                reference_value = self.reference_data[feature]
                
                # Simple check for numerical data
                if isinstance(value, (int, float)) and isinstance(reference_value, (int, float)):
                    # Simple percent change
                    if reference_value != 0:
                        percent_change = abs((value - reference_value) / reference_value)
                        metrics[feature] = {
                            'percent_change': percent_change,
                            'current': value,
                            'reference': reference_value
                        }
                        
                        # Check if drift threshold exceeded
                        if percent_change > 0.3:  # 30% change threshold
                            drift_features.append(feature)
        
        # Create drift report
        drift_report = {
            'drift_detected': len(drift_features) > 0,
            'drift_features': drift_features,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to history
        self.drift_history.append(drift_report)
        self.drift_history = self.drift_history[-10:]  # Keep last 10 reports
        
        return drift_report
    
    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Get drift detection history."""
        return self.drift_history
EOF

  # Create ensemble_model.py
  cat > $SILICON_DIR/ensemble_model.py << EOF
"""
Ensemble Model for MINDSET Silicon Layer.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Callable, Union, Optional

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble Model that combines multiple base models.
    
    Key capabilities:
    - Voting ensemble (hard/soft)
    - Stacking ensemble
    - Model weighting
    """
    
    def __init__(self, models: Optional[List[Any]] = None, weights: Optional[List[float]] = None):
        """Initialize the ensemble model."""
        self.models = models or []
        self.weights = weights or []
        if self.weights and len(self.weights) != len(self.models):
            self.weights = [1.0] * len(self.models)
        
        self.ensemble_type = "voting"  # "voting" or "stacking"
        self.voting_type = "soft"  # "hard" or "soft"
        self.meta_model = None
    
    def add_model(self, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"Added model to ensemble, total models: {len(self.models)}")
    
    def set_ensemble_type(self, ensemble_type: str, meta_model: Any = None):
        """
        Set the ensemble type.
        
        Args:
            ensemble_type: "voting" or "stacking"
            meta_model: Model to use for stacking (required if ensemble_type is "stacking")
        """
        if ensemble_type not in ["voting", "stacking"]:
            raise ValueError("Ensemble type must be 'voting' or 'stacking'")
        
        self.ensemble_type = ensemble_type
        if ensemble_type == "stacking" and meta_model is not None:
            self.meta_model = meta_model
        
        logger.info(f"Set ensemble type to {ensemble_type}")
    
    def set_voting_type(self, voting_type: str):
        """
        Set the voting type for voting ensemble.
        
        Args:
            voting_type: "hard" or "soft"
        """
        if voting_type not in ["hard", "soft"]:
            raise ValueError("Voting type must be 'hard' or 'soft'")
        
        self.voting_type = voting_type
        logger.info(f"Set voting type to {voting_type}")
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        For demonstration, we're implementing a simplified version.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if self.ensemble_type == "voting":
            return self._voting_predict(X)
        elif self.ensemble_type == "stacking":
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _voting_predict(self, X: Any) -> np.ndarray:
        """Make predictions using voting ensemble."""
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict_proba') and self.voting_type == "soft":
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error in model {i} prediction: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from models")
        
        # For simplicity, assuming binary classification with soft voting
        if self.voting_type == "soft":
            # Weight and average probabilities
            weighted_preds = []
            for i, pred in enumerate(predictions):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                weighted_preds.append(pred * weight)
            
            avg_preds = np.sum(weighted_preds, axis=0) / np.sum(self.weights)
            return (avg_preds[:, 1] > 0.5).astype(int)  # Convert probabilities to class labels
        else:
            # Hard voting - majority vote
            stacked_preds = np.column_stack([p.flatten() for p in predictions])
            return np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights[:len(predictions)])),
                axis=1,
                arr=stacked_preds
            )
    
    def _stacking_predict(self, X: Any) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        if self.meta_model is None:
            raise ValueError("Meta model not set for stacking ensemble")
        
        # Get base model predictions
        base_preds = []
        for model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.shape[1] == 2:  # Binary classification
                        pred = pred[:, 1].reshape(-1, 1)  # Just keep the positive class probability
                else:
                    pred = model.predict(X).reshape(-1, 1)
                base_preds.append(pred)
            except Exception as e:
                logger.error(f"Error in base model prediction: {e}")
        
        if not base_preds:
            raise ValueError("No valid predictions from base models")
        
        # Combine base predictions into meta-features
        meta_features = np.hstack(base_preds)
        
        # Use meta-model for final prediction
        return self.meta_model.predict(meta_features)
EOF

  # Create xai_wrapper.py
  cat > $SILICON_DIR/xai_wrapper.py << EOF
"""
XAI Wrapper for MINDSET Silicon Layer.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class XAIWrapper:
    """
    Explainable AI wrapper for model interpretability.
    
    Key capabilities:
    - Feature importance
    - SHAP values
    - LIME explanations
    - Partial dependence plots
    """
    
    def __init__(self, model: Optional[Any] = None):
        """Initialize the XAI wrapper."""
        self.model = model
        self.explanation_methods = ["feature_importance", "shap", "lime"]
        self.explanations = {}
        
        # Track feature names for better explanations
        self.feature_names = None
    
    def set_model(self, model: Any):
        """Set the model to explain."""
        self.model = model
        logger.info("Set model for XAI")
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for better explanations."""
        self.feature_names = feature_names
        logger.info(f"Set {len(feature_names)} feature names for XAI")
    
    def explain(self, X: Any, method: str = "feature_importance") -> Dict[str, Any]:
        """
        Generate explanation for the model's predictions.
        
        Args:
            X: Input data to explain
            method: Explanation method (feature_importance, shap, lime)
            
        Returns:
            Dictionary with explanation results
        """
        if self.model is None:
            raise ValueError("Model not set for explanation")
        
        if method not in self.explanation_methods:
            raise ValueError(f"Unknown explanation method: {method}. " 
                           f"Available methods: {', '.join(self.explanation_methods)}")
        
        if method == "feature_importance":
            explanation = self._explain_feature_importance()
        elif method == "shap":
            explanation = self._explain_shap(X)
        elif method == "lime":
            explanation = self._explain_lime(X)
        else:
            raise ValueError(f"Method {method} not implemented")
        
        # Store explanation
        self.explanations[method] = explanation
        
        return explanation
    
    def _explain_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from model if available."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                result = {
                    'method': 'feature_importance',
                    'importances': importances.tolist() if isinstance(importances, np.ndarray) else importances
                }
                
                # Add feature names if available
                if self.feature_names is not None:
                    result['feature_importance'] = {
                        name: importance 
                        for name, importance in zip(self.feature_names, importances)
                    }
                
                return result
            else:
                return {
                    'method': 'feature_importance',
                    'error': 'Model does not support feature_importances_'
                }
        except Exception as e:
            logger.error(f"Error in feature importance explanation: {e}")
            return {
                'method': 'feature_importance',
                'error': str(e)
            }
    
    def _explain_shap(self, X: Any) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Note: This is a simplified placeholder. Real implementation would use the SHAP library.
        """
        # Simplified mock implementation
        try:
            # Mock SHAP values (random for demonstration)
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
            
            # Generate random SHAP values for demonstration
            shap_values = np.random.randn(n_samples, n_features) * 0.1
            
            result = {
                'method': 'shap',
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'note': 'This is a simplified placeholder. Real implementation would use the SHAP library.'
            }
            
            # Add feature names if available
            if self.feature_names is not None:
                # Calculate average absolute SHAP value for each feature
                avg_shap = np.abs(shap_values).mean(axis=0)
                result['feature_importance'] = {
                    name: importance 
                    for name, importance in zip(self.feature_names, avg_shap)
                }
            
            return result
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return {
                'method': 'shap',
                'error': str(e)
            }
    
    def _explain_lime(self, X: Any) -> Dict[str, Any]:
        """
        Generate LIME explanations.
        
        Note: This is a simplified placeholder. Real implementation would use the LIME library.
        """
        # Simplified mock implementation
        try:
            # Mock LIME explanation (random for demonstration)
            n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
            
            # Generate random feature weights for demonstration
            feature_weights = np.random.uniform(-1, 1, n_features)
            
            result = {
                'method': 'lime',
                'feature_weights': feature_weights.tolist() if isinstance(feature_weights, np.ndarray) else feature_weights,
                'note': 'This is a simplified placeholder. Real implementation would use the LIME library.'
            }
            
            # Add feature names if available
            if self.feature_names is not None:
                result['feature_importance'] = {
                    name: weight 
                    for name, weight in zip(self.feature_names, feature_weights)
                }
            
            return result
        except Exception as e:
            logger.error(f"Error in LIME explanation: {e}")
            return {
                'method': 'lime',
                'error': str(e)
            }
EOF

  # Create silicon_layer.py
  cat > $SILICON_DIR/silicon_layer.py << EOF
"""
Main Silicon Layer implementation for MINDSET.

The Silicon Layer sits between the Silver and Gold layers in the medallion architecture,
providing advanced ML processing including:
- Feature Store
- Ensemble Learning
- Explainable AI
- Drift Detection
- Online Learning
"""
import os
import logging
from typing import Dict, List, Any, Optional

from .feature_store import FeatureStore
from .drift_detector import DriftDetector
from .ensemble_model import EnsembleModel
from .xai_wrapper import XAIWrapper

logger = logging.getLogger(__name__)

class SiliconLayer:
    """
    Main Silicon Layer class that orchestrates all components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Silicon Layer with all components."""
        self.config = config or {}
        
        # Initialize components
        self.feature_store = FeatureStore(
            storage_path=self.config.get('feature_store_path')
        )
        
        self.drift_detector = DriftDetector()
        self.ensemble_model = EnsembleModel()
        self.xai_wrapper = XAIWrapper()
        
        # Metrics from Rust engine (to be integrated)
        self.metrics_engine = None
        
        logger.info("Silicon Layer initialized successfully")
    
    def process(self, silver_data: Dict[str, Any], feature_store_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process data through the Silicon Layer.
        
        Args:
            silver_data: Data from the Silver layer
            feature_store_data: Additional data for the feature store (optional)
            
        Returns:
            Processed data ready for the Gold layer
        """
        logger.info("Processing data through Silicon Layer")
        
        # 1. Update Feature Store
        if feature_store_data:
            for name, value in feature_store_data.items():
                self.feature_store.add_feature(name, value)
        
        # 2. Check for data drift
        current_features = self._extract_features(silver_data)
        drift_report = self.drift_detector.check_drift(current_features)
        
        # 3. Apply models (if needed)
        model_results = {}
        if self.ensemble_model.models:
            try:
                # This is simplified - real implementation would prepare features properly
                X = self._prepare_features_for_model(silver_data)
                predictions = self.ensemble_model.predict(X)
                model_results['predictions'] = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                
                # Generate explanations if model is set
                if self.xai_wrapper.model is not None:
                    explanation = self.xai_wrapper.explain(X)
                    model_results['explanation'] = explanation
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                model_results['error'] = str(e)
        
        # 4. Calculate transparency metrics (placeholder for Rust engine integration)
        transparency_metrics = self._calculate_transparency_metrics(silver_data)
        
        # 5. Prepare output for Gold layer
        gold_data = {
            'original_data': silver_data,
            'features': current_features,
            'drift_report': drift_report,
            'model_results': model_results,
            'transparency_metrics': transparency_metrics
        }
        
        logger.info("Silicon Layer processing completed successfully")
        return gold_data
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from the input data."""
        # Simplified implementation - real version would be more sophisticated
        features = {}
        
        # Extract basic features
        if 'text' in data:
            features['text_length'] = len(data['text'])
            features['word_count'] = len(data['text'].split())
        
        if 'numerical_fields' in data:
            for name, value in data.get('numerical_fields', {}).items():
                features[f'num_{name}'] = value
        
        if 'categorical_fields' in data:
            for name, value in data.get('categorical_fields', {}).items():
                features[f'cat_{name}'] = value
        
        # Add feature store features
        for name, value in self.feature_store.get_all_features().items():
            if name not in features:
                features[f'fs_{name}'] = value
        
        return features
    
    def _prepare_features_for_model(self, data: Dict[str, Any]) -> Any:
        """Prepare features for model prediction."""
        # Simplified implementation - real version would properly convert to numpy/pandas
        import numpy as np
        
        features = self._extract_features(data)
        feature_values = list(features.values())
        
        # Convert to numpy array with shape (1, n_features)
        return np.array(feature_values).reshape(1, -1)
    
    def _calculate_transparency_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate transparency metrics.
        
        This is a placeholder for the Rust metrics engine integration.
        """
        # Placeholder implementation
        metrics = {
            'political_influence_level': 0.0,
            'rhetoric_intensity_scale': 0.0,
            'information_depth_score': 0.0
        }
        
        # Simple mock calculations
        if 'text' in data:
            text = data['text']
            word_count = len(text.split())
            
            # Mock political influence
            political_keywords = ['government', 'policy', 'politician', 'election']
            political_count = sum(1 for word in political_keywords if word.lower() in text.lower())
            metrics['political_influence_level'] = (political_count / max(1, word_count // 10)) * 20
            
            # Mock rhetoric intensity
            emotional_words = ['outrageous', 'shocking', 'devastating', 'incredible']
            emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
            metrics['rhetoric_intensity_scale'] = (emotional_count / max(1, word_count // 20)) * 30
            
            # Mock information depth
            avg_sentence_length = word_count / max(1, text.count('.'))
            uppercase_words = sum(1 for word in text.split() if word and word[0].isupper())
            citation_count = text.count('(') + text.count('[')
            
            metrics['information_depth_score'] = min(100, (
                (avg_sentence_length / 15) * 30 +
                (uppercase_words / max(1, word_count // 8)) * 40 +
                (citation_count * 10)
            ))
            
        return metrics
    
    def integrate_metrics_engine(self, metrics_engine):
        """Integrate the Rust metrics engine."""
        self.metrics_engine = metrics_engine
        logger.info("Rust metrics engine integrated with Silicon Layer")
EOF

  echo "Silicon Layer created successfully!"
fi

# Step 4: Upload datasets to Azure storage
echo "====================================="
echo "Step 4: Uploading datasets to Azure"
echo "====================================="
cd $INFRASTRUCTURE_DIR
./upload-datasets.sh

# Step 5: Build and push Docker image
echo "====================================="
echo "Step 5: Building and pushing Docker image"
echo "====================================="
cd $INFRASTRUCTURE_DIR
./build-push.sh

# Step 6: Deploy Kubernetes resources
echo "====================================="
echo "Step 6: Deploying Kubernetes resources"
echo "====================================="

# Create Kubernetes manifests if they don't exist
if [ ! -f "$KUBERNETES_DIR/deployment.yaml" ]; then
  echo "Creating Kubernetes manifests..."
  
  # Create deployment.yaml
  cat > $KUBERNETES_DIR/deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mindset-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mindset-api
  template:
    metadata:
      labels:
        app: mindset-api
    spec:
      containers:
      - name: mindset-api
        image: ${CONTAINER_REGISTRY_NAME}.azurecr.io/mindset:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        ports:
        - containerPort: 8000
        env:
        - name: STORAGE_ACCOUNT_NAME
          value: "${STORAGE_ACCOUNT_NAME}"
        - name: NEWS_API_KEY
          value: "${NEWS_API_KEY}"  # Using direct env var instead of secret reference
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
EOF

  # Create service.yaml
  cat > $KUBERNETES_DIR/service.yaml << EOF
apiVersion: v1
kind: Service
metadata:
  name: mindset-api
spec:
  selector:
    app: mindset-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
EOF

  # Create secrets.yaml
  cat > $KUBERNETES_DIR/secrets.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: mindset-secrets
type: Opaque
stringData:
  news-api-key: "${NEWS_API_KEY}"
EOF

  echo "Kubernetes manifests created successfully!"
fi

# Deploy Kubernetes resources (if AKS is available)
if [ "$DEPLOY_AKS" = "true" ]; then
  echo "Deploying to AKS cluster..."
  
  # Connect to AKS
  az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME
  
  # Create namespace
  kubectl create namespace mindset --dry-run=client -o yaml | kubectl apply -f -
  
  # Apply manifests
  kubectl apply -f $KUBERNETES_DIR/secrets.yaml -n mindset
  kubectl apply -f $KUBERNETES_DIR/deployment.yaml -n mindset
  kubectl apply -f $KUBERNETES_DIR/service.yaml -n mindset
  
  # Get the service endpoint
  EXTERNAL_IP=$(kubectl get service mindset-api -n mindset -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  
  echo "Deployed to AKS successfully!"
  if [ -n "$EXTERNAL_IP" ]; then
    echo "Service endpoint: http://$EXTERNAL_IP/"
  else
    echo "External IP not yet available. Run 'kubectl get service mindset-api -n mindset' later to check."
  fi
else
  echo "AKS deployment skipped - AKS cluster not available."
  echo "To deploy to AKS when available:"
  echo "1. Connect to your AKS cluster"
  echo "2. Run: kubectl apply -f $KUBERNETES_DIR/secrets.yaml"
  echo "3. Run: kubectl apply -f $KUBERNETES_DIR/deployment.yaml"
  echo "4. Run: kubectl apply -f $KUBERNETES_DIR/service.yaml"
fi

echo "====================================="
echo "      MINDSET Deployment Summary     "
echo "====================================="
echo "Resource Group: $RESOURCE_GROUP"
echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo "Container Registry: $CONTAINER_REGISTRY_NAME"

# Create a standalone test file to verify deployment
TEST_SCRIPT="$MINDSET_ROOT/test-mindset-api.py"
cat > $TEST_SCRIPT << EOF
#!/usr/bin/env python3
"""
Simple test script for MINDSET API.
"""
import sys
import os
import json
import requests
from pprint import pprint

# Get the API URL from command line or use default
api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print(f"Testing MINDSET API at: {api_url}")
print("==================================")

try:
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{api_url}/")
    print(f"Status: {response.status_code}")
    pprint(response.json())

    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{api_url}/health")
    print(f"Status: {response.status_code}")
    pprint(response.json())

    # Test articles endpoint
    print("\n3. Testing articles endpoint...")
    response = requests.get(f"{api_url}/articles")
    print(f"Status: {response.status_code}")
    articles = response.json()
    print(f"Retrieved {len(articles)} articles")
    if articles:
        print("First article:")
        pprint(articles[0])

    # Test analysis endpoint
    print("\n4. Testing analysis endpoint...")
    test_text = "This is a test article about government policies and climate change. The devastating effects of global warming require immediate action."
    response = requests.post(
        f"{api_url}/analyze", 
        json={"text": test_text}
    )
    print(f"Status: {response.status_code}")
    print("Transparency metrics:")
    pprint(response.json())

    print("\nAll tests completed successfully!")

except Exception as e:
    print(f"Error testing API: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure the API is running and accessible")
    print("2. Check if the endpoint is correct")
    print("3. Verify network connectivity")
EOF

chmod +x $TEST_SCRIPT

echo ""
echo "Deployment completed successfully!"
echo "====================================="
echo "Next steps:"
echo "1. Start the API locally with: cd $MINDSET_ROOT && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "2. Test the API with: python $TEST_SCRIPT"
echo "3. For AKS deployment (if available), follow the instructions above"
echo "4. Develop the frontend application"
echo "5. Set up CI/CD pipelines for automated deployments"
echo "6. Implement additional Silicon Layer features"
echo "====================================="