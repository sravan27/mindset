"""
Silicon Layer - Advanced ML processing layer between Silver and Gold
Main module for the Silicon Layer
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

# Import silicon layer components
from .feature_store import FeatureStore
from .drift_detector import DriftDetector
from .ensemble_model import MetricsEnsembleModel, create_default_metrics_ensemble
from .xai_wrapper import MetricsExplainer

class SiliconLayer:
    """
    Silicon Layer for advanced ML processing
    
    Functions:
    - Feature storage and versioning
    - Drift detection
    - Ensemble model training and prediction
    - Explainable AI
    - Metrics calculation (Political Influence, Rhetoric Intensity, Information Depth)
    """
    
    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        use_feature_store: bool = True,
        use_drift_detection: bool = True,
        use_ensemble_models: bool = True,
        use_xai: bool = True,
        metrics_engine: str = 'python',  # 'python' or 'rust'
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Silicon Layer
        
        Args:
            base_dir: Base directory for the Silicon Layer
            use_feature_store: Whether to use the feature store
            use_drift_detection: Whether to use drift detection
            use_ensemble_models: Whether to use ensemble models
            use_xai: Whether to use explainable AI
            metrics_engine: Engine for metrics calculation ('python' or 'rust')
            logger: Logger instance
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / 'silicon_layer'
        self.metrics_engine = metrics_engine
        
        # Create directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logger or self._setup_logger()
        
        # Initialize components
        self.feature_store = None
        if use_feature_store:
            self.feature_store = FeatureStore(base_dir=self.base_dir / 'feature_store')
            self.logger.info("Feature store initialized")
        
        self.drift_detector = None
        if use_drift_detection:
            self.drift_detector = DriftDetector(metrics_dir=self.base_dir / 'drift_metrics')
            self.logger.info("Drift detector initialized")
        
        self.ensemble_model = None
        if use_ensemble_models:
            self.ensemble_model = create_default_metrics_ensemble(models_dir=self.base_dir / 'models')
            self.logger.info("Ensemble model initialized")
        
        self.explainer = None
        if use_xai and self.ensemble_model:
            # Will be initialized after models are fitted
            self.logger.info("XAI ready for initialization (after model training)")
        
        # Try to initialize Rust metrics engine if requested
        self.rust_engine = None
        if metrics_engine == 'rust':
            try:
                import metrics_engine_rust
                self.rust_engine = metrics_engine_rust
                self.logger.info("Rust metrics engine initialized")
            except ImportError:
                self.logger.warning("Rust metrics engine not available. Falling back to Python implementation.")
                self.metrics_engine = 'python'
        
        self.logger.info(f"Silicon Layer initialized with {metrics_engine} metrics engine")
    
    def _setup_logger(self):
        """Set up logger for the Silicon Layer"""
        logger = logging.getLogger('silicon_layer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        
        # Create file handler if directory exists
        if self.base_dir:
            log_dir = self.base_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'silicon_layer.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def process(self, data: Union[pd.DataFrame, List[Dict]]) -> Union[pd.DataFrame, List[Dict]]:
        """
        Process data through the Silicon Layer
        
        Args:
            data: Data to process (DataFrame or list of dictionaries)
            
        Returns:
            Processed data with added metrics
        """
        self.logger.info(f"Processing data through Silicon Layer: {len(data)} items")
        
        # Convert list of dictionaries to DataFrame if necessary
        input_is_dict_list = isinstance(data, list) and all(isinstance(item, dict) for item in data)
        if input_is_dict_list:
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Store features if feature store is enabled
        if self.feature_store:
            try:
                # Store only relevant features (excluding metrics, etc.)
                feature_cols = [col for col in df.columns if col not in ['metrics', 'id', 'url']]
                if feature_cols:
                    features_df = df[feature_cols].copy()
                    version = self.feature_store.store_features(
                        features_df,
                        feature_name='article_features',
                        description='News article features for metrics calculation'
                    )
                    self.logger.info(f"Features stored with version {version}")
            except Exception as e:
                self.logger.error(f"Error storing features: {e}")
        
        # Check for drift if drift detector is enabled and has reference data
        if self.drift_detector and hasattr(self.drift_detector, 'reference_data') and self.drift_detector.reference_data is not None:
            try:
                # Only use numeric columns for drift detection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    drift_metrics = self.drift_detector.calculate_drift_metrics(df, columns=numeric_cols)
                    drift_summary = self.drift_detector.get_drift_summary(drift_metrics)
                    
                    if drift_summary['drift_percentage'] > 0.3:  # If more than 30% of features show drift
                        self.logger.warning(f"Significant drift detected: {drift_summary['drift_percentage']:.2%} of features")
                        # In a real system, you might trigger alerts or model retraining
            except Exception as e:
                self.logger.error(f"Error detecting drift: {e}")
        
        # Calculate metrics
        processed_data = self._calculate_metrics(df)
        
        # Convert back to list of dictionaries if input was list
        if input_is_dict_list:
            if isinstance(processed_data, pd.DataFrame):
                processed_data = processed_data.to_dict('records')
        
        self.logger.info(f"Data processing complete: {len(processed_data)} items processed")
        return processed_data
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics for articles
        
        Args:
            df: DataFrame with article data
            
        Returns:
            DataFrame with added metrics
        """
        self.logger.info(f"Calculating metrics for {len(df)} articles")
        
        # Initialize metrics columns if they don't exist
        if 'metrics' not in df.columns:
            df['metrics'] = [{}] * len(df)
        
        # Choose calculation method based on selected engine
        if self.metrics_engine == 'rust' and self.rust_engine:
            df = self._calculate_metrics_rust(df)
        else:
            df = self._calculate_metrics_python(df)
        
        return df
    
    def _calculate_metrics_rust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics using Rust engine
        
        Args:
            df: DataFrame with article data
            
        Returns:
            DataFrame with added metrics
        """
        self.logger.info("Using Rust engine for metrics calculation")
        
        try:
            # Process each article
            for i, row in df.iterrows():
                # Extract text content for analysis
                content = row.get('content', '')
                if not content and 'abstract' in row:
                    content = row['abstract']
                if not content and 'title' in row:
                    content = row['title']
                
                if not content:
                    # Skip if no content available
                    continue
                
                # Call Rust function to calculate metrics
                political_influence, rhetoric_intensity, information_depth = self.rust_engine.calculate_metrics(content)
                
                # Ensure metrics dictionary exists
                if df.at[i, 'metrics'] is None or not isinstance(df.at[i, 'metrics'], dict):
                    df.at[i, 'metrics'] = {}
                
                # Add metrics
                df.at[i, 'metrics']['political_influence'] = political_influence
                df.at[i, 'metrics']['rhetoric_intensity'] = rhetoric_intensity
                df.at[i, 'metrics']['information_depth'] = information_depth
        
        except Exception as e:
            self.logger.error(f"Error using Rust metrics engine: {e}")
            self.logger.info("Falling back to Python implementation")
            return self._calculate_metrics_python(df)
        
        return df
    
    def _calculate_metrics_python(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics using Python implementation
        
        Args:
            df: DataFrame with article data
            
        Returns:
            DataFrame with added metrics
        """
        self.logger.info("Using Python implementation for metrics calculation")
        
        # Check if we have a trained model
        if self.ensemble_model and hasattr(self.ensemble_model, 'political_influence_model') and self.ensemble_model.political_influence_model is not None:
            # Use trained model
            try:
                # Prepare features
                features = self._extract_features(df)
                
                # Get predictions from model
                predictions = self.ensemble_model.predict(features)
                
                # Add predictions to DataFrame
                for i, row in df.iterrows():
                    # Ensure metrics dictionary exists
                    if df.at[i, 'metrics'] is None or not isinstance(df.at[i, 'metrics'], dict):
                        df.at[i, 'metrics'] = {}
                    
                    # Add metrics
                    for metric, values in predictions.items():
                        df.at[i, 'metrics'][metric] = float(values[i])
            
            except Exception as e:
                self.logger.error(f"Error using trained model for metrics: {e}")
                self.logger.info("Falling back to rule-based implementation")
                return self._calculate_metrics_rule_based(df)
        else:
            # No trained model, use rule-based approach
            return self._calculate_metrics_rule_based(df)
        
        return df
    
    def _calculate_metrics_rule_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics using simple rule-based approach
        
        Args:
            df: DataFrame with article data
            
        Returns:
            DataFrame with added metrics
        """
        self.logger.info("Using rule-based approach for metrics calculation")
        
        import re
        from collections import Counter
        import random
        
        # Political words (simplified)
        political_words = [
            'government', 'president', 'congress', 'senate', 'election', 'vote', 'democrat', 'republican',
            'liberal', 'conservative', 'policy', 'regulation', 'law', 'senator', 'representative',
            'party', 'campaign', 'politician', 'administration', 'bill', 'legislation'
        ]
        
        # Rhetoric words (simplified)
        rhetoric_words = [
            'must', 'never', 'always', 'outrageous', 'incredible', 'terrible', 'amazing', 'worst',
            'best', 'catastrophic', 'crisis', 'disaster', 'triumph', 'victory', 'defeat', 'scandal',
            'shocking', 'bombshell', 'explosive', 'stunning', 'devastating', 'tremendous'
        ]
        
        # Information depth indicators (simplified)
        depth_indicators = [
            'according to', 'research', 'study', 'data', 'evidence', 'analysis', 'survey', 'experts',
            'statistics', 'report', 'investigation', 'findings', 'concluded', 'discovered', 'determined',
            'interviewed', 'documented', 'confirmed', 'verified', 'sources'
        ]
        
        for i, row in df.iterrows():
            # Extract text content for analysis
            content = row.get('content', '')
            if not content and 'abstract' in row:
                content = row['abstract']
            if not content and 'title' in row:
                content = row['title']
            
            if not content:
                # Skip if no content available
                continue
            
            # Ensure metrics dictionary exists
            if df.at[i, 'metrics'] is None or not isinstance(df.at[i, 'metrics'], dict):
                df.at[i, 'metrics'] = {}
            
            # Convert to lowercase for analysis
            content_lower = content.lower()
            
            # Count instances of indicator words
            word_counts = Counter(re.findall(r'\b\w+\b', content_lower))
            
            # Calculate political influence
            political_count = sum(word_counts.get(word, 0) for word in political_words)
            # Normalize to 0-10 scale (simplified)
            political_influence = min(10, political_count / 2)
            
            # Calculate rhetoric intensity
            rhetoric_count = sum(word_counts.get(word, 0) for word in rhetoric_words)
            # Normalize to 0-10 scale (simplified)
            rhetoric_intensity = min(10, rhetoric_count * 1.5)
            
            # Calculate information depth
            depth_count = sum(1 for indicator in depth_indicators if indicator in content_lower)
            # Normalize to 0-10 scale (simplified)
            information_depth = min(10, depth_count * 1.5)
            
            # Add some randomness to make it more realistic
            political_influence = max(0, min(10, political_influence + random.uniform(-0.5, 0.5)))
            rhetoric_intensity = max(0, min(10, rhetoric_intensity + random.uniform(-0.5, 0.5)))
            information_depth = max(0, min(10, information_depth + random.uniform(-0.5, 0.5)))
            
            # Add metrics
            df.at[i, 'metrics']['political_influence'] = float(political_influence)
            df.at[i, 'metrics']['rhetoric_intensity'] = float(rhetoric_intensity)
            df.at[i, 'metrics']['information_depth'] = float(information_depth)
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from articles for model prediction
        
        Args:
            df: DataFrame with article data
            
        Returns:
            NumPy array of features
        """
        # In a real implementation, this would extract NLP features
        # For now, return a simple feature matrix based on text length and word counts
        
        import re
        from collections import Counter
        
        features = []
        
        for _, row in df.iterrows():
            # Extract text content
            content = row.get('content', '')
            if not content and 'abstract' in row:
                content = row['abstract']
            if not content and 'title' in row:
                content = row['title']
            
            if not content:
                # Use empty values if no content
                features.append([0, 0, 0, 0, 0])
                continue
            
            # Basic features
            content_lower = content.lower()
            word_count = len(re.findall(r'\b\w+\b', content_lower))
            avg_word_length = sum(len(word) for word in re.findall(r'\b\w+\b', content_lower)) / max(1, word_count)
            
            # Count sentence endings as a proxy for number of sentences
            sentence_count = len(re.findall(r'[.!?]', content))
            
            # Count question marks and exclamation points
            question_count = content.count('?')
            exclamation_count = content.count('!')
            
            features.append([
                word_count,
                avg_word_length,
                sentence_count,
                question_count,
                exclamation_count
            ])
        
        return np.array(features)
    
    def train(self, training_data: pd.DataFrame, target_columns: Dict[str, str]):
        """
        Train the ensemble model
        
        Args:
            training_data: DataFrame with training data
            target_columns: Dictionary mapping metric names to target column names
            
        Returns:
            self
        """
        self.logger.info(f"Training ensemble model with {len(training_data)} samples")
        
        if not self.ensemble_model:
            self.ensemble_model = create_default_metrics_ensemble(models_dir=self.base_dir / 'models')
        
        try:
            # Extract features
            X = self._extract_features(training_data)
            
            # Extract targets
            y_dict = {}
            for metric_name, column_name in target_columns.items():
                if column_name in training_data.columns:
                    y_dict[metric_name] = training_data[column_name].values
            
            # Train the model
            self.ensemble_model.fit(X, y_dict)
            
            # Initialize explainer after training
            if not self.explainer:
                self.explainer = MetricsExplainer(
                    models_dict={
                        'political_influence': self.ensemble_model.political_influence_model,
                        'rhetoric_intensity': self.ensemble_model.rhetoric_intensity_model,
                        'information_depth': self.ensemble_model.information_depth_model
                    },
                    feature_names=['word_count', 'avg_word_length', 'sentence_count', 'question_count', 'exclamation_count'],
                    explanations_dir=self.base_dir / 'explanations'
                )
                self.explainer.create_explainers(X[:min(100, len(X))])
            
            # Save the trained model
            if self.ensemble_model.models_dir:
                path = self.ensemble_model.save()
                self.logger.info(f"Model saved to {path}")
            
            # Set training data as reference for drift detection
            if self.drift_detector:
                self.drift_detector.set_reference_data(training_data)
            
            self.logger.info("Model training completed successfully")
        
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
        
        return self
    
    def load_model(self, path: Union[str, Path]):
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
            
        Returns:
            self
        """
        self.logger.info(f"Loading model from {path}")
        
        try:
            self.ensemble_model = MetricsEnsembleModel.load(path)
            
            # Initialize explainer after loading
            self.explainer = MetricsExplainer(
                models_dict={
                    'political_influence': self.ensemble_model.political_influence_model,
                    'rhetoric_intensity': self.ensemble_model.rhetoric_intensity_model,
                    'information_depth': self.ensemble_model.information_depth_model
                },
                feature_names=['word_count', 'avg_word_length', 'sentence_count', 'question_count', 'exclamation_count'],
                explanations_dir=self.base_dir / 'explanations'
            )
            
            self.logger.info("Model loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        return self
    
    def explain_article(self, article: Union[Dict, pd.Series, int], data: Optional[pd.DataFrame] = None):
        """
        Generate explanations for an article
        
        Args:
            article: Article to explain (dictionary, Series, or index in data)
            data: DataFrame containing the article (required if article is an index)
            
        Returns:
            Explanation dictionary
        """
        if not self.explainer:
            self.logger.warning("Explainer not initialized. Train or load a model first.")
            return {"error": "Explainer not initialized"}
        
        try:
            # Convert article to DataFrame for feature extraction
            if isinstance(article, int) and data is not None:
                article_df = pd.DataFrame([data.iloc[article]])
                article_idx = 0
            elif isinstance(article, pd.Series):
                article_df = pd.DataFrame([article])
                article_idx = 0
            elif isinstance(article, dict):
                article_df = pd.DataFrame([article])
                article_idx = 0
            else:
                raise ValueError("Article must be a dictionary, Series, or index")
            
            # Extract features
            X = self._extract_features(article_df)
            
            # Generate explanations
            explanation = self.explainer.explain_article(X, idx=article_idx)
            
            return explanation
        
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return {"error": str(e)}