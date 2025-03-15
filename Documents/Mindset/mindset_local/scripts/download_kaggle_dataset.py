#!/usr/bin/env python3
"""
Kaggle News Category Dataset Downloader
Downloads and extracts the Kaggle News Category Dataset for MINDSET.
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: Kaggle API package not found. Install with:")
    print("pip install kaggle")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not found. Install with:")
    print("pip install pandas")
    sys.exit(1)

def load_kaggle_credentials(config_file=None):
    """Load Kaggle credentials from config or environment."""
    # Try to load from config file
    if config_file and os.path.exists(config_file):
        sys.path.append(os.path.dirname(config_file))
        from api_config import KAGGLE_USERNAME, KAGGLE_KEY
        return KAGGLE_USERNAME, KAGGLE_KEY
    
    # Check if credentials are already in environment
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if username and key:
        return username, key
    
    # Final fallback: check for kaggle.json
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_json):
        with open(kaggle_json, 'r') as f:
            creds = json.load(f)
            return creds.get('username'), creds.get('key')
    
    print("Error: Kaggle credentials not found.")
    print("Either:")
    print("1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    print("2. Create ~/.kaggle/kaggle.json with your credentials")
    print("3. Update config/api_config.py with your credentials")
    sys.exit(1)

def download_dataset(username, key, target_dir):
    """Download the Kaggle News Category Dataset."""
    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    # Create target directory
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dataset ID for the Kaggle News Category Dataset
    dataset = "rmisra/news-category-dataset"
    
    print(f"Downloading {dataset} to {target_dir}...")
    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    print("Download complete!")
    
    # Verify download
    json_file = target_dir / "News_Category_Dataset_v3.json"
    if not json_file.exists():
        json_file = target_dir / "News_Category_Dataset_v2.json"
        if not json_file.exists():
            print("Error: Expected JSON file not found after download.")
            sys.exit(1)
    
    # Basic dataset info
    try:
        with open(json_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"Downloaded dataset with {line_count} articles")
        
        # Convert to sample DataFrame for quick verification
        df = pd.read_json(json_file, lines=True)
        print(f"Dataset categories: {df['category'].nunique()} unique categories")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        print(f"Warning: Could not read dataset for verification: {e}")
    
    return json_file

def main():
    """Main function to download the Kaggle News Category Dataset."""
    parser = argparse.ArgumentParser(description="Download Kaggle News Category Dataset")
    parser.add_argument(
        "--config", 
        help="Path to configuration file containing Kaggle credentials",
        default=None
    )
    parser.add_argument(
        "--target", 
        help="Target directory to download the dataset",
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine base directory and config location
    if args.config is None:
        base_dir = Path(__file__).parent.parent
        config_file = base_dir / "config" / "api_config.py"
        if not config_file.exists():
            config_file = None
    else:
        config_file = args.config
    
    # Determine target directory
    if args.target is None:
        base_dir = Path(__file__).parent.parent
        target_dir = base_dir / "data" / "raw" / "kaggle_news"
    else:
        target_dir = Path(args.target)
    
    # Load credentials and download
    username, key = load_kaggle_credentials(config_file)
    json_file = download_dataset(username, key, target_dir)
    
    print(f"\nDataset downloaded and extracted to: {target_dir}")
    print(f"Main dataset file: {json_file}")
    print("\nNext step: Process the dataset with process_datasets.py")

if __name__ == "__main__":
    main()