#!/usr/bin/env python3
"""
Dataset Processing Script for MINDSET
Processes raw datasets (MIND and Kaggle News) into structured formats for the Bronze layer.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import csv
import random
import multiprocessing

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
except ImportError:
    raise ImportError("Required libraries not found. Please run setup_environment.py first.")

# Set up constants
MIND_DATASET_FILENAME = "news.tsv"
MIND_BEHAVIOR_FILENAME = "behaviors.tsv"
KAGGLE_NEWS_FILENAME_V3 = "News_Category_Dataset_v3.json"
KAGGLE_NEWS_FILENAME_V2 = "News_Category_Dataset_v2.json"

def get_kaggle_news_filename(data_dir):
    """Get the Kaggle news dataset filename based on what's available."""
    v3_path = os.path.join(data_dir, KAGGLE_NEWS_FILENAME_V3)
    v2_path = os.path.join(data_dir, KAGGLE_NEWS_FILENAME_V2)
    
    if os.path.exists(v3_path):
        return v3_path
    elif os.path.exists(v2_path):
        return v2_path
    else:
        raise FileNotFoundError(f"Kaggle News dataset not found in {data_dir}")

def process_mind_news(mind_data_dir, output_dir, split_name, sample_size=None):
    """Process MIND news.tsv into a structured parquet file."""
    news_path = os.path.join(mind_data_dir, MIND_DATASET_FILENAME)
    
    if not os.path.exists(news_path):
        print(f"Warning: {news_path} not found. Skipping MIND news processing for {split_name}.")
        return None
    
    print(f"Processing MIND news data from {news_path}...")
    
    # Define column names for the MIND news dataset
    columns = [
        "news_id", "category", "subcategory", "title", "abstract", 
        "url", "entity", "relation"
    ]
    
    # Read the TSV file with correct encoding
    df = pd.read_csv(
        news_path, 
        sep='\t', 
        names=columns, 
        quoting=csv.QUOTE_NONE, 
        encoding='utf-8'
    )
    
    # Take a sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Process entity field (it's a JSON string)
    def parse_entity(entity_str):
        if pd.isna(entity_str) or entity_str == '[]':
            return []
        try:
            return json.loads(entity_str)
        except:
            return []
    
    df['entity'] = df['entity'].apply(parse_entity)
    
    # Extract entity information
    df['entity_count'] = df['entity'].apply(len)
    df['entity_types'] = df['entity'].apply(
        lambda entities: [e.get('Type', '') for e in entities]
    )
    df['entity_labels'] = df['entity'].apply(
        lambda entities: [e.get('Label', '') for e in entities]
    )
    
    # Add dataset metadata
    df['source'] = 'MIND'
    df['split'] = split_name
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, f"mind_{split_name}_news.parquet")
    
    # Write to parquet format
    df.to_parquet(output_path, index=False)
    
    print(f"Processed {len(df)} MIND news articles to {output_path}")
    return output_path

def process_mind_behaviors(mind_data_dir, output_dir, split_name, sample_size=None):
    """Process MIND behaviors.tsv into a structured parquet file."""
    behaviors_path = os.path.join(mind_data_dir, MIND_BEHAVIOR_FILENAME)
    
    if not os.path.exists(behaviors_path):
        print(f"Warning: {behaviors_path} not found. Skipping MIND behaviors processing for {split_name}.")
        return None
    
    print(f"Processing MIND behaviors data from {behaviors_path}...")
    
    # Define column names for the MIND behaviors dataset
    columns = [
        "impression_id", "user_id", "time", "history", "impressions"
    ]
    
    # Read the TSV file with correct encoding
    df = pd.read_csv(
        behaviors_path, 
        sep='\t', 
        names=columns, 
        quoting=csv.QUOTE_NONE, 
        encoding='utf-8'
    )
    
    # Take a sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Process history field (comma-separated news IDs)
    df['history'] = df['history'].apply(
        lambda x: [] if pd.isna(x) else x.split()
    )
    df['history_length'] = df['history'].apply(len)
    
    # Process impressions field
    def process_impressions(x):
        if pd.isna(x):
            return []
        
        result = []
        for item in x.split():
            parts = item.split('-')
            if len(parts) >= 2:
                result.append({
                    'news_id': parts[0],
                    'clicked': parts[1] == '1'
                })
            else:
                # Handle impression items without click information
                result.append({
                    'news_id': parts[0],
                    'clicked': False  # Default to not clicked
                })
        return result
    
    df['impressions'] = df['impressions'].apply(process_impressions)
    df['impression_count'] = df['impressions'].apply(len)
    
    # Add click count
    df['click_count'] = df['impressions'].apply(
        lambda impressions: sum(1 for imp in impressions if imp['clicked'])
    )
    
    # Add dataset metadata
    df['source'] = 'MIND'
    df['split'] = split_name
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, f"mind_{split_name}_behaviors.parquet")
    
    # Write to parquet format
    df.to_parquet(output_path, index=False)
    
    print(f"Processed {len(df)} MIND behavior records to {output_path}")
    return output_path

def process_kaggle_news(kaggle_data_dir, output_dir, sample_size=None):
    """Process Kaggle News dataset into a structured parquet file."""
    try:
        news_path = get_kaggle_news_filename(kaggle_data_dir)
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping Kaggle news processing.")
        return None
    
    print(f"Processing Kaggle news data from {news_path}...")
    
    # Read the JSON file (line-delimited)
    df = pd.read_json(news_path, lines=True)
    
    # Take a sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Standardize column names
    rename_map = {
        'headline': 'title',
        'short_description': 'abstract',
        'link': 'url',
        'authors': 'author',
        'date': 'published_date'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Ensure consistent date format
    if 'published_date' in df.columns:
        df['published_date'] = pd.to_datetime(df['published_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add source identifier
    df['source'] = 'Kaggle'
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Generate a unique ID for each article
    df['news_id'] = [f"K{i}" for i in range(len(df))]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, "kaggle_news.parquet")
    
    # Write to parquet format
    df.to_parquet(output_path, index=False)
    
    print(f"Processed {len(df)} Kaggle news articles to {output_path}")
    return output_path

def create_dask_dataset(parquet_files, output_dir):
    """Create a Dask dataset from parquet files for parallel processing."""
    if not parquet_files:
        print("Warning: No parquet files provided. Skipping Dask dataset creation.")
        return None
    
    print(f"Creating Dask dataset from {len(parquet_files)} parquet files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read each parquet file first to identify common columns
    print("Identifying common columns across datasets...")
    common_columns = None
    for file in parquet_files:
        df = pd.read_parquet(file)
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns = common_columns.intersection(set(df.columns))
    
    common_columns = list(common_columns)
    print(f"Found {len(common_columns)} common columns across all datasets")
    
    # Read parquet files into a Dask dataframe, selecting only common columns
    ddf = dd.read_parquet(parquet_files, columns=common_columns)
    
    # Define output path for the Dask dataset
    output_path = os.path.join(output_dir, "news_dataset")
    
    # Write the Dask dataframe to parquet (partitioned by source)
    print("Writing unified dataset to parquet files...")
    with ProgressBar():
        ddf.to_parquet(
            output_path, 
            engine='pyarrow',
            compression='snappy',
            partition_on=['source'] if 'source' in common_columns else None,
            write_index=False
        )
    
    print(f"Created Dask dataset at {output_path}")
    return output_path

def generate_sample_dataset(input_dirs, output_dir, sample_size=1000):
    """Generate a small sample dataset for testing and development."""
    all_files = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue
        for file in os.listdir(input_dir):
            if file.endswith('.parquet'):
                all_files.append(os.path.join(input_dir, file))
    
    if not all_files:
        print("Warning: No parquet files found. Cannot generate sample dataset.")
        return None
    
    print(f"Generating sample dataset with {sample_size} records...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and sample from each file
    samples = []
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            file_sample_size = min(sample_size // len(all_files), len(df))
            samples.append(df.sample(file_sample_size, random_state=42))
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")
    
    if not samples:
        print("Warning: Could not read any files. Cannot generate sample dataset.")
        return None
    
    # Combine samples
    combined = pd.concat(samples, ignore_index=True)
    
    # Define output path for the sample dataset
    output_path = os.path.join(output_dir, "sample_dataset.parquet")
    
    # Write to parquet format
    combined.to_parquet(output_path, index=False)
    
    print(f"Created sample dataset with {len(combined)} records at {output_path}")
    return output_path

def main():
    """Main function to process all datasets."""
    parser = argparse.ArgumentParser(description="Process raw datasets for MINDSET")
    parser.add_argument("--base-dir", help="Base directory for MINDSET", default=None)
    parser.add_argument("--sample", type=int, help="Sample size for each dataset", default=None)
    parser.add_argument("--create-test-sample", action="store_true", help="Create a small sample dataset")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(args.base_dir)
    
    # Define data directories
    raw_dir = base_dir / "data" / "raw"
    bronze_dir = base_dir / "data" / "bronze"
    
    # Define MIND dataset directories
    mind_train_dir = raw_dir / "mind_train"
    mind_dev_dir = raw_dir / "mind_dev"
    mind_test_dir = raw_dir / "mind_test"
    
    # Define Kaggle dataset directory
    kaggle_dir = raw_dir / "kaggle_news"
    
    # Create output directories
    bronze_dir.mkdir(exist_ok=True, parents=True)
    
    # Process MIND datasets
    processed_files = []
    
    # Process MIND train
    if mind_train_dir.exists():
        train_news = process_mind_news(mind_train_dir, bronze_dir, "train", args.sample)
        train_behaviors = process_mind_behaviors(mind_train_dir, bronze_dir, "train", args.sample)
        if train_news:
            processed_files.append(train_news)
    
    # Process MIND dev
    if mind_dev_dir.exists():
        dev_news = process_mind_news(mind_dev_dir, bronze_dir, "dev", args.sample)
        dev_behaviors = process_mind_behaviors(mind_dev_dir, bronze_dir, "dev", args.sample)
        if dev_news:
            processed_files.append(dev_news)
    
    # Process MIND test
    if mind_test_dir.exists():
        test_news = process_mind_news(mind_test_dir, bronze_dir, "test", args.sample)
        test_behaviors = process_mind_behaviors(mind_test_dir, bronze_dir, "test", args.sample)
        if test_news:
            processed_files.append(test_news)
    
    # Process Kaggle news
    if kaggle_dir.exists():
        kaggle_news = process_kaggle_news(kaggle_dir, bronze_dir, args.sample)
        if kaggle_news:
            processed_files.append(kaggle_news)
    
    # Create Dask dataset
    if processed_files:
        dask_dataset = create_dask_dataset(processed_files, bronze_dir / "dask")
    
    # Generate sample dataset if requested
    if args.create_test_sample:
        sample_dataset = generate_sample_dataset([bronze_dir], bronze_dir / "sample", 1000)
    
    print("\nDataset processing complete!")
    print(f"Processed files available in: {bronze_dir}")
    print("\nNext step: Run feature engineering with silver_layer.py")

if __name__ == "__main__":
    main()