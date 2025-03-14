#!/usr/bin/env python
"""
Train ML models for the Silicon Layer
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Silicon Layer
from silicon_layer.silicon_layer import SiliconLayer
from silicon_layer.ensemble_model import EnsembleModel, MetricsEnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_training')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SILICON_DATA_DIR = DATA_DIR / "silicon"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
SILICON_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def generate_training_data(size=1000, seed=42):
    """
    Generate synthetic training data for metrics models
    
    Args:
        size: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with features and target metrics
    """
    logger.info(f"Generating {size} synthetic training samples")
    
    np.random.seed(seed)
    
    # Generate features
    data = []
    
    for i in range(size):
        # Text features
        word_count = np.random.randint(50, 1500)  # Number of words
        avg_word_length = np.random.normal(5, 1)  # Average word length
        sentence_count = max(1, int(word_count / np.random.randint(10, 25)))  # Number of sentences
        question_count = np.random.poisson(1)  # Number of questions
        exclamation_count = np.random.poisson(0.5)  # Number of exclamations
        
        # Calculate metrics with some noise
        # 1. Political Influence (0-10)
        # - Higher word count slightly increases political content
        # - More questions may indicate less political bias
        base_political = np.random.beta(2, 3) * 10  # Base distribution centered lower
        word_count_effect = word_count / 5000 * 2  # 0-0.6 effect
        question_effect = -question_count * 0.2  # Negative effect
        political_influence = max(0, min(10, base_political + word_count_effect + question_effect + np.random.normal(0, 1)))
        
        # 2. Rhetoric Intensity (0-10)
        # - More exclamations increase rhetoric
        # - Shorter sentences increase rhetoric
        base_rhetoric = np.random.beta(2, 4) * 10  # Base distribution centered lower
        exclamation_effect = exclamation_count * 0.5  # Positive effect
        sentence_effect = 3 - (sentence_count / word_count * 100)  # Effect for short sentences
        rhetoric_intensity = max(0, min(10, base_rhetoric + exclamation_effect + sentence_effect + np.random.normal(0, 0.8)))
        
        # 3. Information Depth (0-10)
        # - Longer words increase depth
        # - More words generally increase depth
        base_depth = np.random.beta(3, 2) * 10  # Base distribution centered higher
        word_length_effect = (avg_word_length - 4) * 0.8  # Effect for longer words
        word_count_effect = word_count / 1000 * 2  # 0-3 effect
        information_depth = max(0, min(10, base_depth + word_length_effect + word_count_effect + np.random.normal(0, 0.7)))
        
        # Create sample
        sample = {
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'political_influence': political_influence,
            'rhetoric_intensity': rhetoric_intensity,
            'information_depth': information_depth
        }
        
        data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def train_models(training_data, test_size=0.2, random_state=42):
    """
    Train ML models for metrics prediction
    
    Args:
        training_data: DataFrame with features and target metrics
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Trained MetricsEnsembleModel
    """
    logger.info("Training models for metrics prediction")
    
    # Define features and targets
    feature_cols = ['word_count', 'avg_word_length', 'sentence_count', 'question_count', 'exclamation_count']
    target_cols = ['political_influence', 'rhetoric_intensity', 'information_depth']
    
    # Split data
    X = training_data[feature_cols].values
    y_dict = {target: training_data[target].values for target in target_cols}
    
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    for target in target_cols:
        X_train[target], X_test[target], y_train_dict[target], y_test_dict[target] = train_test_split(
            X, y_dict[target], test_size=test_size, random_state=random_state
        )
    
    # Create ensemble model
    metrics_ensemble = MetricsEnsembleModel(models_dir=MODELS_DIR)
    
    # Create and train models for each metric
    for metric in target_cols:
        logger.info(f"Training model for {metric}")
        
        # Create base models
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        
        # Train base models
        rf.fit(X_train[metric], y_train_dict[metric])
        gb.fit(X_train[metric], y_train_dict[metric])
        
        # Create ensemble
        ensemble = EnsembleModel(
            ensemble_type='voting',
            models=[rf, gb],
            weights=[0.5, 0.5],
            models_dir=MODELS_DIR / metric
        )
        
        # Set ensemble for the metric
        if metric == 'political_influence':
            metrics_ensemble.set_political_influence_model(ensemble)
        elif metric == 'rhetoric_intensity':
            metrics_ensemble.set_rhetoric_intensity_model(ensemble)
        elif metric == 'information_depth':
            metrics_ensemble.set_information_depth_model(ensemble)
        
        # Evaluate
        y_pred = ensemble.predict(X_test[metric])
        rmse = np.sqrt(mean_squared_error(y_test_dict[metric], y_pred))
        r2 = r2_score(y_test_dict[metric], y_pred)
        
        logger.info(f"{metric} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save full ensemble
    ensemble_path = metrics_ensemble.save("metrics_ensemble")
    logger.info(f"Ensemble saved to {ensemble_path}")
    
    return metrics_ensemble

def train_and_integrate():
    """
    Train models and integrate with Silicon Layer
    """
    logger.info("Starting model training and integration")
    
    # Generate training data
    training_data = generate_training_data(size=2000)
    
    # Train models
    metrics_ensemble = train_models(training_data)
    
    # Initialize Silicon Layer
    silicon_layer = SiliconLayer(
        base_dir=SILICON_DATA_DIR,
        use_feature_store=True,
        use_drift_detection=True,
        use_ensemble_models=True,
        use_xai=True,
        metrics_engine='python'
    )
    
    # Integrate trained model
    silicon_layer.ensemble_model = metrics_ensemble
    
    # Save sample data for testing
    sample_size = min(100, len(training_data))
    sample_data = []
    
    for i in range(sample_size):
        # Create article-like structure
        article = {
            'news_id': f"sample_{i}",
            'title': f"Sample Article {i}",
            'abstract': f"This is a sample abstract for article {i}.",
            'content': f"This is the content of sample article {i}.",
            'url': f"https://example.com/article/{i}",
            'category': np.random.choice(['news', 'politics', 'business', 'technology', 'science']),
            'source': 'Sample Source',
            'published_at': pd.Timestamp.now().isoformat(),
            'metrics': {
                'political_influence': float(training_data.iloc[i]['political_influence']),
                'rhetoric_intensity': float(training_data.iloc[i]['rhetoric_intensity']),
                'information_depth': float(training_data.iloc[i]['information_depth'])
            }
        }
        
        sample_data.append(article)
    
    # Save sample data
    sample_path = DATA_DIR / "sample_articles.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Saved {len(sample_data)} sample articles to {sample_path}")
    
    logger.info("Model training and integration complete")

def main():
    parser = argparse.ArgumentParser(description="Train ML models for the Silicon Layer")
    parser.add_argument("--synthetic-size", type=int, default=2000,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()
    
    train_and_integrate()

if __name__ == "__main__":
    main()