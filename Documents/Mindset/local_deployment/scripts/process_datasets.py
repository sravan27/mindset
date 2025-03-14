#!/usr/bin/env python
"""
Process datasets through the medallion architecture (Raw -> Bronze -> Silver -> Silicon -> Gold)
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Union, Optional
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Silicon Layer
from silicon_layer.silicon_layer import SiliconLayer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_pipeline')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
SILICON_DATA_DIR = DATA_DIR / "silicon"
GOLD_DATA_DIR = DATA_DIR / "gold"

# Ensure data directories exist
for dir_path in [RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, SILICON_DATA_DIR, GOLD_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def load_mind_dataset(dataset_size: str = "small", split: str = "train") -> pd.DataFrame:
    """
    Load MIND dataset
    
    Args:
        dataset_size: Size of the dataset ("small" or "large")
        split: Split to load ("train", "dev", or "test")
        
    Returns:
        DataFrame with news articles
    """
    logger.info(f"Loading MIND{dataset_size}_{split} dataset")
    
    # Paths
    dataset_path = RAW_DATA_DIR / "MIND" / f"MIND{dataset_size}_{split}"
    news_path = dataset_path / "news.tsv"
    
    if not news_path.exists():
        logger.error(f"News file not found at {news_path}")
        return pd.DataFrame()
    
    # Load news data
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    )
    
    logger.info(f"Loaded {len(news_df)} articles from MIND{dataset_size}_{split}")
    return news_df

def load_newsapi_articles() -> pd.DataFrame:
    """
    Load articles from NewsAPI
    
    Returns:
        DataFrame with news articles
    """
    logger.info("Loading NewsAPI articles")
    
    # Paths
    newsapi_dir = RAW_DATA_DIR / "newsapi"
    
    if not newsapi_dir.exists():
        logger.warning(f"NewsAPI directory not found at {newsapi_dir}")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = list(newsapi_dir.glob("*.json"))
    
    if not json_files:
        logger.warning("No NewsAPI article files found")
        return pd.DataFrame()
    
    articles = []
    
    # Load all JSON files
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Add articles to list
            articles.extend(data)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not articles:
        logger.warning("No articles found in NewsAPI files")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Rename columns to match MIND dataset
    column_mapping = {
        'title': 'title',
        'description': 'abstract',
        'url': 'url',
        'source.name': 'source_name',
        'publishedAt': 'published_at',
        'content': 'content'
    }
    
    # Select and rename columns
    result_df = pd.DataFrame()
    for new_col, old_col in column_mapping.items():
        if '.' in old_col:
            # Handle nested fields
            parts = old_col.split('.')
            if parts[0] in df.columns and isinstance(df[parts[0]].iloc[0], dict):
                result_df[new_col] = df[parts[0]].apply(lambda x: x.get(parts[1], '') if x else '')
        elif old_col in df.columns:
            result_df[new_col] = df[old_col]
    
    # Add news_id
    result_df['news_id'] = [f"newsapi_{i}" for i in range(len(result_df))]
    
    # Add category and subcategory
    result_df['category'] = 'news'
    result_df['subcategory'] = 'general'
    
    logger.info(f"Loaded {len(result_df)} articles from NewsAPI")
    return result_df

def process_raw_to_bronze() -> pd.DataFrame:
    """
    Process data from Raw to Bronze layer
    - Load data from multiple sources
    - Clean and standardize the data
    - Deduplicate articles
    
    Returns:
        Bronze DataFrame
    """
    logger.info("Processing Raw → Bronze")
    
    # Load MIND datasets
    mind_small_train = load_mind_dataset(dataset_size="small", split="train")
    mind_small_dev = load_mind_dataset(dataset_size="small", split="dev")
    
    # Load NewsAPI articles
    newsapi_df = load_newsapi_articles()
    
    # Combine datasets
    dfs = []
    if not mind_small_train.empty:
        mind_small_train['source'] = 'mind_small_train'
        dfs.append(mind_small_train)
    
    if not mind_small_dev.empty:
        mind_small_dev['source'] = 'mind_small_dev'
        dfs.append(mind_small_dev)
    
    if not newsapi_df.empty:
        newsapi_df['source'] = 'newsapi'
        dfs.append(newsapi_df)
    
    if not dfs:
        logger.error("No data available for processing")
        return pd.DataFrame()
    
    # Combine all datasets
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Combined data: {len(df)} articles")
    
    # Clean and standardize the data
    bronze_df = clean_data(df)
    
    # Save to Bronze layer
    bronze_path = BRONZE_DATA_DIR / f"articles_{datetime.now().strftime('%Y%m%d')}.parquet"
    table = pa.Table.from_pandas(bronze_df)
    pq.write_table(table, bronze_path)
    
    logger.info(f"Saved {len(bronze_df)} articles to Bronze layer: {bronze_path}")
    
    return bronze_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the data
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy
    clean_df = df.copy()
    
    # Fill missing values
    clean_df['title'] = clean_df['title'].fillna('')
    clean_df['abstract'] = clean_df['abstract'].fillna('')
    clean_df['url'] = clean_df['url'].fillna('')
    clean_df['category'] = clean_df['category'].fillna('unknown')
    clean_df['subcategory'] = clean_df['subcategory'].fillna('unknown')
    
    # Remove duplicates based on title
    clean_df = clean_df.drop_duplicates(subset=['title'], keep='first')
    
    # Remove articles with empty title or very short content
    clean_df = clean_df[(clean_df['title'].str.len() > 5)]
    
    # Clean URLs
    clean_df['url'] = clean_df['url'].str.strip()
    
    return clean_df

def process_bronze_to_silver(bronze_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Process data from Bronze to Silver layer
    - Extract features from text
    - Convert categorical variables
    - Apply transformations
    
    Args:
        bronze_df: Bronze DataFrame (if None, load from Bronze layer)
        
    Returns:
        Silver DataFrame
    """
    logger.info("Processing Bronze → Silver")
    
    # Load from Bronze layer if not provided
    if bronze_df is None:
        # Find the latest Bronze file
        bronze_files = list(BRONZE_DATA_DIR.glob("*.parquet"))
        if not bronze_files:
            logger.error("No Bronze files found")
            return pd.DataFrame()
        
        latest_bronze_file = max(bronze_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading from Bronze layer: {latest_bronze_file}")
        
        bronze_df = pd.read_parquet(latest_bronze_file)
    
    # Apply feature engineering
    silver_df = engineer_features(bronze_df)
    
    # Save to Silver layer
    silver_path = SILVER_DATA_DIR / f"articles_{datetime.now().strftime('%Y%m%d')}.parquet"
    table = pa.Table.from_pandas(silver_df)
    pq.write_table(table, silver_path)
    
    logger.info(f"Saved {len(silver_df)} articles to Silver layer: {silver_path}")
    
    return silver_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the Silver layer
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy
    silver_df = df.copy()
    
    # Extract date from published_at if available
    if 'published_at' in silver_df.columns:
        try:
            silver_df['published_date'] = pd.to_datetime(silver_df['published_at']).dt.date
        except Exception as e:
            logger.warning(f"Error extracting date from published_at: {e}")
    
    # Text features
    silver_df['title_length'] = silver_df['title'].str.len()
    
    if 'abstract' in silver_df.columns:
        silver_df['abstract_length'] = silver_df['abstract'].str.len()
    
    # Count words
    silver_df['title_word_count'] = silver_df['title'].str.split().str.len()
    
    if 'abstract' in silver_df.columns:
        silver_df['abstract_word_count'] = silver_df['abstract'].str.split().str.len()
    
    # One-hot encode categories
    if 'category' in silver_df.columns:
        category_dummies = pd.get_dummies(silver_df['category'], prefix='cat')
        silver_df = pd.concat([silver_df, category_dummies], axis=1)
    
    return silver_df

def process_silver_to_silicon(silver_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Process data from Silver to Silicon layer
    - Apply ML models
    - Calculate metrics
    - Add explanations
    
    Args:
        silver_df: Silver DataFrame (if None, load from Silver layer)
        
    Returns:
        Silicon DataFrame
    """
    logger.info("Processing Silver → Silicon")
    
    # Load from Silver layer if not provided
    if silver_df is None:
        # Find the latest Silver file
        silver_files = list(SILVER_DATA_DIR.glob("*.parquet"))
        if not silver_files:
            logger.error("No Silver files found")
            return pd.DataFrame()
        
        latest_silver_file = max(silver_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading from Silver layer: {latest_silver_file}")
        
        silver_df = pd.read_parquet(latest_silver_file)
    
    # Initialize Silicon Layer
    silicon_layer = SiliconLayer(
        base_dir=SILICON_DATA_DIR,
        use_feature_store=True,
        use_drift_detection=True,
        use_ensemble_models=True,
        use_xai=True,
        metrics_engine='python'
    )
    
    # Process through Silicon Layer
    silicon_df = silicon_layer.process(silver_df)
    
    # Save to Silicon layer
    silicon_path = SILICON_DATA_DIR / f"articles_{datetime.now().strftime('%Y%m%d')}.parquet"
    table = pa.Table.from_pandas(silicon_df)
    pq.write_table(table, silicon_path)
    
    logger.info(f"Saved {len(silicon_df)} articles to Silicon layer: {silicon_path}")
    
    return silicon_df

def process_silicon_to_gold(silicon_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Process data from Silicon to Gold layer
    - Add recommendations
    - Prepare for final consumption
    
    Args:
        silicon_df: Silicon DataFrame (if None, load from Silicon layer)
        
    Returns:
        Gold DataFrame
    """
    logger.info("Processing Silicon → Gold")
    
    # Load from Silicon layer if not provided
    if silicon_df is None:
        # Find the latest Silicon file
        silicon_files = list(SILICON_DATA_DIR.glob("*.parquet"))
        if not silicon_files:
            logger.error("No Silicon files found")
            return pd.DataFrame()
        
        latest_silicon_file = max(silicon_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading from Silicon layer: {latest_silicon_file}")
        
        silicon_df = pd.read_parquet(latest_silicon_file)
    
    # Prepare Gold layer
    gold_df = prepare_gold_data(silicon_df)
    
    # Save to Gold layer
    gold_path = GOLD_DATA_DIR / f"articles_{datetime.now().strftime('%Y%m%d')}.parquet"
    table = pa.Table.from_pandas(gold_df)
    pq.write_table(table, gold_path)
    
    logger.info(f"Saved {len(gold_df)} articles to Gold layer: {gold_path}")
    
    return gold_df

def prepare_gold_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Gold layer
    
    Args:
        df: DataFrame to process
        
    Returns:
        Gold DataFrame
    """
    # Create a copy
    gold_df = df.copy()
    
    # Add recommendations
    gold_df['recommendations'] = [[] for _ in range(len(gold_df))]
    
    # Generate recommendations (simplified approach)
    if len(gold_df) > 1:
        # For each article, find similar articles based on category
        for i, row in gold_df.iterrows():
            category = row.get('category', 'unknown')
            
            # Find articles with the same category
            similar_indices = gold_df[gold_df['category'] == category].index.tolist()
            
            # Remove the current article
            if i in similar_indices:
                similar_indices.remove(i)
            
            # Take up to 5 recommendations
            rec_indices = similar_indices[:5]
            
            # Add to recommendations
            gold_df.at[i, 'recommendations'] = [
                {
                    'news_id': gold_df.at[idx, 'news_id'],
                    'title': gold_df.at[idx, 'title'],
                    'score': 1.0 - (0.1 * j)  # Simple score calculation
                }
                for j, idx in enumerate(rec_indices)
            ]
    
    # Select columns for final output
    columns_to_keep = [
        'news_id', 'title', 'abstract', 'url', 'category', 'subcategory',
        'source', 'published_at', 'metrics', 'recommendations'
    ]
    
    # Keep only columns that exist
    final_columns = [col for col in columns_to_keep if col in gold_df.columns]
    gold_df = gold_df[final_columns]
    
    return gold_df

def main():
    parser = argparse.ArgumentParser(description="Process datasets through medallion architecture")
    parser.add_argument("--start-layer", choices=["raw", "bronze", "silver", "silicon"], default="raw",
                        help="Starting layer for processing")
    parser.add_argument("--end-layer", choices=["bronze", "silver", "silicon", "gold"], default="gold",
                        help="Ending layer for processing")
    args = parser.parse_args()
    
    # Determine which layers to process
    layers_order = ["raw", "bronze", "silver", "silicon", "gold"]
    start_idx = layers_order.index(args.start_layer)
    end_idx = layers_order.index(args.end_layer) + 1
    layers_to_process = layers_order[start_idx:end_idx]
    
    logger.info(f"Processing layers: {' → '.join(layers_to_process)}")
    
    # Process each layer
    current_df = None
    
    if "raw" in layers_to_process and "bronze" in layers_to_process:
        current_df = process_raw_to_bronze()
    
    if "bronze" in layers_to_process and "silver" in layers_to_process:
        current_df = process_bronze_to_silver(current_df)
    
    if "silver" in layers_to_process and "silicon" in layers_to_process:
        current_df = process_silver_to_silicon(current_df)
    
    if "silicon" in layers_to_process and "gold" in layers_to_process:
        current_df = process_silicon_to_gold(current_df)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()