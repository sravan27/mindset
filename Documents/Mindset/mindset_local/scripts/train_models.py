#!/usr/bin/env python3
"""
MINDSET Model Training Script
Trains and evaluates models for news metrics prediction.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
except ImportError:
    raise ImportError("Required libraries not found. Please run setup_environment.py first.")

# Add parent directory to path for local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from silicon_layer.silicon_layer import SiliconLayer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"mindset_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('mindset.train_models')

def load_training_data(data_path, sample_size=None):
    """Load and prepare training data."""
    logger.info(f"Loading training data from: {data_path}")
    
    # Check if path exists
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Load the data
    try:
        df = pd.read_parquet(data_path)
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} records from {len(df)} total records")
            df = df.sample(sample_size, random_state=42)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description="Train MINDSET models")
    parser.add_argument("--base-dir", help="Base directory for MINDSET", default=None)
    parser.add_argument("--data-path", help="Path to training data", default=None)
    parser.add_argument("--sample", type=int, help="Sample size for training", default=None)
    parser.add_argument("--test-size", type=float, help="Fraction of data to use for testing", default=0.2)
    parser.add_argument("--skip-explainers", action="store_true", help="Skip creating explainers")
    parser.add_argument("--skip-drift", action="store_true", help="Skip setting up drift detection")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(args.base_dir)
    
    # Determine data path
    if args.data_path is None:
        data_path = base_dir / "data" / "silver" / "mind_train_news_features.parquet"
        if not data_path.exists():
            # Try alternative paths
            alt_paths = [
                base_dir / "data" / "silver" / "kaggle_news_features.parquet",
                base_dir / "data" / "silver" / "mind_dev_news_features.parquet",
                base_dir / "data" / "bronze" / "sample" / "sample_dataset.parquet"
            ]
            
            for path in alt_paths:
                if path.exists():
                    data_path = path
                    break
    else:
        data_path = Path(args.data_path)
    
    try:
        # Load training data
        training_data = load_training_data(data_path, args.sample)
        
        # Initialize Silicon Layer
        silicon_layer = SiliconLayer(
            base_dir=str(base_dir),
            data_dir=str(base_dir / "data"),
            model_dir=str(base_dir / "data" / "silicon_layer" / "models"),
            explainer_dir=str(base_dir / "data" / "silicon_layer" / "explainers"),
            drift_dir=str(base_dir / "data" / "silicon_layer" / "drift"),
            random_state=42
        )
        
        # Train models
        training_results = silicon_layer.train_models(
            training_data,
            test_size=args.test_size,
            create_explainers=not args.skip_explainers,
            setup_drift_detector=not args.skip_drift
        )
        
        # Export models for API
        exported_models = silicon_layer.export_models_for_api()
        
        # Generate performance report
        report_path = base_dir / "data" / "silicon_layer" / "reports" / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True, parents=True)
        performance_report = silicon_layer.generate_performance_report(str(report_path))
        
        logger.info("Model training and evaluation complete")
        logger.info(f"Performance report saved to: {report_path}")
        logger.info(f"Models exported to: {base_dir / 'data' / 'gold' / 'models'}")
        
        # Print summary of results
        print("\n========== TRAINING SUMMARY ==========")
        print(f"Data source: {data_path}")
        print(f"Records: {len(training_data)}")
        
        print("\nModel Evaluation:")
        for target, metrics in performance_report["evaluation_metrics"].items():
            print(f"  {target}:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    R²:   {metrics['r2']:.4f}")
        
        print("\nTop Important Features:")
        if "feature_importance" in performance_report and performance_report["feature_importance"]:
            first_metric = list(performance_report["feature_importance"].keys())[0]
            features = list(performance_report["feature_importance"][first_metric].items())[:5]
            for feature, importance in features:
                print(f"  {feature}: {importance:.4f}")
        
        print("\nExported Models:")
        for name, path in exported_models.items():
            print(f"  {name}: {path}")
        
        print("\nFor detailed results, see:")
        print(f"  Performance report: {report_path}")
        print(f"  Log file: mindset_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        print("=======================================")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()