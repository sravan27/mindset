#!/usr/bin/env python3
"""
Feature Engineering Script for MINDSET
Creates and transforms features for the Silver layer using Rust acceleration.
"""

import os
import json
import argparse
from pathlib import Path
import multiprocessing
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from tqdm import tqdm
except ImportError:
    raise ImportError("Required libraries not found. Please run setup_environment.py first.")

# Try to import Rust acceleration module
try:
    import mindset_rust
    RUST_AVAILABLE = True
    print("Using Rust acceleration for feature engineering")
except ImportError:
    RUST_AVAILABLE = False
    print("Rust acceleration not available. Using pure Python implementation.")

# Load political and emotional term dictionaries
def load_term_dictionaries(config_dir):
    """Load term dictionaries for feature calculation."""
    # Load or create political terms
    political_terms_path = os.path.join(config_dir, "political_terms.json")
    emotional_terms_path = os.path.join(config_dir, "emotional_terms.json")
    
    # Create default dictionaries if files don't exist
    if not os.path.exists(political_terms_path):
        # Sample political terms with weights
        political_terms = {
            "government": 0.3,
            "president": 0.4,
            "election": 0.5,
            "congress": 0.4,
            "democrat": 0.7,
            "republican": 0.7,
            "policy": 0.3,
            "liberal": 0.6,
            "conservative": 0.6,
            "political": 0.4,
            "legislation": 0.3,
            "vote": 0.5,
            "campaign": 0.5,
            "candidate": 0.5,
            "partisan": 0.8,
            "bipartisan": 0.6,
            "scandal": 0.7,
            "administration": 0.4,
            "senate": 0.4,
            "house of representatives": 0.4,
            "left wing": 0.7,
            "right wing": 0.7,
            "progressive": 0.6,
            "socialist": 0.7,
            "nationalist": 0.7,
            "populist": 0.6,
            "constitution": 0.4,
            "amendment": 0.4,
            "supreme court": 0.5,
            "judicial": 0.4,
            "law": 0.3,
            "regulation": 0.3,
            "bill": 0.3,
            "tax": 0.4,
            "budget": 0.3,
            "deficit": 0.4,
            "debt": 0.3,
            "economy": 0.3,
            "healthcare": 0.4,
            "immigration": 0.5,
            "climate change": 0.5,
            "foreign policy": 0.4,
            "national security": 0.5,
            "military": 0.4,
            "war": 0.6
        }
        
        # Save default dictionary
        os.makedirs(os.path.dirname(political_terms_path), exist_ok=True)
        with open(political_terms_path, 'w') as f:
            json.dump(political_terms, f, indent=2)
    else:
        # Load existing dictionary
        with open(political_terms_path, 'r') as f:
            political_terms = json.load(f)
    
    # Create or load emotional terms
    if not os.path.exists(emotional_terms_path):
        # Sample emotional terms with intensity scores
        emotional_terms = {
            "amazing": 0.7,
            "terrible": 0.8,
            "excited": 0.6,
            "furious": 0.9,
            "happy": 0.5,
            "sad": 0.5,
            "angry": 0.8,
            "thrilled": 0.7,
            "devastated": 0.9,
            "wonderful": 0.6,
            "horrible": 0.8,
            "awesome": 0.6,
            "awful": 0.7,
            "excellent": 0.6,
            "dreadful": 0.7,
            "love": 0.6,
            "hate": 0.8,
            "thrilling": 0.7,
            "catastrophic": 0.9,
            "shocking": 0.8,
            "outrageous": 0.8,
            "absurd": 0.7,
            "ridiculous": 0.7,
            "fantastic": 0.6,
            "disaster": 0.8,
            "tragic": 0.8,
            "incredible": 0.6,
            "disgrace": 0.8,
            "unbelievable": 0.7,
            "terrifying": 0.8,
            "tremendous": 0.6,
            "horrific": 0.9,
            "stunning": 0.6,
            "appalling": 0.8,
            "marvelous": 0.6,
            "catastrophe": 0.9,
            "delightful": 0.5,
            "disastrous": 0.8,
            "exceptional": 0.6,
            "disgusting": 0.8,
            "spectacular": 0.7,
            "dire": 0.7,
            "extraordinary": 0.6,
            "astonishing": 0.7,
            "devastating": 0.8
        }
        
        # Save default dictionary
        os.makedirs(os.path.dirname(emotional_terms_path), exist_ok=True)
        with open(emotional_terms_path, 'w') as f:
            json.dump(emotional_terms, f, indent=2)
    else:
        # Load existing dictionary
        with open(emotional_terms_path, 'r') as f:
            emotional_terms = json.load(f)
    
    return political_terms, emotional_terms

# Pure Python implementations for when Rust is not available
def py_process_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    """Python implementation of text processing."""
    if not isinstance(text, str):
        return ""
    
    result = text.lower() if lowercase else text
    
    if remove_punctuation:
        import string
        translator = str.maketrans('', '', string.punctuation.replace("'", ""))
        result = result.translate(translator)
    
    return result

def py_calculate_political_influence(text: str, political_terms: Dict[str, float]) -> float:
    """Calculate political influence score in pure Python."""
    if not isinstance(text, str) or not text:
        return 0.0
        
    lowercase_text = text.lower()
    words = lowercase_text.split()
    
    score = 0.0
    term_count = 0
    
    for term, weight in political_terms.items():
        term_lowercase = term.lower()
        term_words = term_lowercase.split()
        
        if len(term_words) == 1:
            # Single word term
            occurrences = words.count(term_words[0])
            score += weight * occurrences
            term_count += occurrences
        else:
            # Multi-word term (phrase)
            # Simple string search for phrases
            if term_lowercase in lowercase_text:
                score += weight
                term_count += 1
    
    # Normalize score
    if term_count == 0:
        return 0.0
    
    return min(score / term_count, 1.0)

def py_calculate_rhetoric_intensity(text: str, emotional_terms: Dict[str, float]) -> float:
    """Calculate rhetoric intensity score in pure Python."""
    if not isinstance(text, str) or not text:
        return 0.0
        
    lowercase_text = text.lower()
    words = lowercase_text.split()
    
    total_intensity = 0.0
    term_count = 0
    
    for term, intensity in emotional_terms.items():
        term_lowercase = term.lower()
        
        # Count occurrences
        occurrences = words.count(term_lowercase)
        total_intensity += intensity * occurrences
        term_count += occurrences
    
    # Normalize
    if term_count == 0:
        return 0.0
    
    return min(total_intensity / term_count, 1.0)

def py_calculate_information_depth(text: str) -> float:
    """Calculate information depth score in pure Python."""
    if not isinstance(text, str) or not text:
        return 0.0
        
    # Word count
    words = text.split()
    count = len(words)
    
    # Vocabulary diversity
    unique_words = set(words)
    unique_ratio = len(unique_words) / count if count > 0 else 0.0
    
    # Citations check
    has_citations = "(" in text and ")" in text
    citation_bonus = 0.2 if has_citations else 0.0
    
    # Calculate factors
    length_factor = min(count / 800, 1.0)
    complexity_factor = min(unique_ratio, 0.8)
    
    # Combine into final score
    raw_score = 0.5 * length_factor + 0.3 * complexity_factor + citation_bonus
    
    return min(max(raw_score, 0.0), 1.0)

def py_calculate_article_metrics(
    text: str, 
    political_terms: Dict[str, float],
    emotional_terms: Dict[str, float]
) -> tuple:
    """Calculate all metrics in one pass (Python implementation)."""
    if not isinstance(text, str) or not text:
        return (0.0, 0.0, 0.0)
        
    political = py_calculate_political_influence(text, political_terms)
    rhetoric = py_calculate_rhetoric_intensity(text, emotional_terms)
    depth = py_calculate_information_depth(text)
    
    return (political, rhetoric, depth)

def process_article(article: Dict[str, Any], political_terms: Dict[str, float], emotional_terms: Dict[str, float]) -> Dict[str, Any]:
    """Process a single article to extract features."""
    # Copy the original article
    processed = article.copy()
    
    # Get the text content (combine title and abstract for analysis)
    title = article.get('title', '')
    abstract = article.get('abstract', '')
    content = f"{title} {abstract}"
    
    # Skip empty content
    if not content.strip():
        processed['political_influence'] = 0.0
        processed['rhetoric_intensity'] = 0.0
        processed['information_depth'] = 0.0
        processed['information_depth_category'] = 'Overview'
        return processed
    
    # Process text using available implementation
    if RUST_AVAILABLE:
        political, rhetoric, depth = mindset_rust.calculate_article_metrics(
            content, 
            political_terms, 
            emotional_terms
        )
    else:
        political, rhetoric, depth = py_calculate_article_metrics(
            content,
            political_terms,
            emotional_terms
        )
    
    # Add metrics to processed article
    processed['political_influence'] = float(political)
    processed['rhetoric_intensity'] = float(rhetoric)
    processed['information_depth'] = float(depth)
    
    # Categorize information depth
    if depth < 0.33:
        processed['information_depth_category'] = 'Overview'
    elif depth < 0.67:
        processed['information_depth_category'] = 'Analysis'
    else:
        processed['information_depth_category'] = 'In-depth'
    
    return processed

def process_dataframe(df: pd.DataFrame, political_terms: Dict[str, float], emotional_terms: Dict[str, float]) -> pd.DataFrame:
    """Process an entire dataframe of articles."""
    # Create a list for processed articles
    processed_articles = []
    
    # Process each article
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        processed = process_article(row.to_dict(), political_terms, emotional_terms)
        processed_articles.append(processed)
    
    # Create new dataframe with processed articles
    return pd.DataFrame(processed_articles)

def process_parquet_file(
    input_file: str, 
    output_file: str, 
    political_terms: Dict[str, float], 
    emotional_terms: Dict[str, float],
    sample_size: Optional[int] = None
) -> str:
    """Process a single parquet file and save the results."""
    # Read the parquet file
    df = pd.read_parquet(input_file)
    
    # Take a sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Process the dataframe
    processed_df = process_dataframe(df, political_terms, emotional_terms)
    
    # Save the processed dataframe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_df.to_parquet(output_file, index=False)
    
    print(f"Processed {len(processed_df)} articles from {input_file} to {output_file}")
    return output_file

def process_dask_dataset(
    input_dir: str, 
    output_dir: str, 
    political_terms: Dict[str, float], 
    emotional_terms: Dict[str, float],
    n_workers: int = None
) -> str:
    """Process a Dask dataset in parallel."""
    if not os.path.exists(input_dir):
        print(f"Warning: Input directory {input_dir} not found. Skipping Dask processing.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up number of workers for Dask
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Processing Dask dataset with {n_workers} workers...")
    
    # Define a Dask processing function
    def process_partition(partition):
        """Process a partition of the Dask DataFrame."""
        return process_dataframe(partition, political_terms, emotional_terms)
    
    # Read the dataset
    ddf = dd.read_parquet(input_dir)
    
    # Process the dataset
    with ProgressBar():
        result = ddf.map_partitions(process_partition).persist()
        
        # Write the result
        result.to_parquet(
            output_dir,
            engine='pyarrow',
            compression='snappy',
            write_index=False
        )
    
    print(f"Processed Dask dataset from {input_dir} to {output_dir}")
    return output_dir

def main():
    """Main function to run feature engineering."""
    parser = argparse.ArgumentParser(description="Feature Engineering for MINDSET")
    parser.add_argument("--base-dir", help="Base directory for MINDSET", default=None)
    parser.add_argument("--sample", type=int, help="Sample size for processing", default=None)
    parser.add_argument("--workers", type=int, help="Number of workers for parallel processing", default=None)
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(args.base_dir)
    
    # Define data directories
    bronze_dir = base_dir / "data" / "bronze"
    silver_dir = base_dir / "data" / "silver"
    config_dir = base_dir / "config"
    
    # Create output directories
    silver_dir.mkdir(exist_ok=True, parents=True)
    config_dir.mkdir(exist_ok=True, parents=True)
    
    # Load term dictionaries
    political_terms, emotional_terms = load_term_dictionaries(config_dir)
    
    # Print status
    print(f"Loaded {len(political_terms)} political terms and {len(emotional_terms)} emotional terms")
    
    # Process individual parquet files
    for split in ["train", "dev", "test"]:
        input_file = bronze_dir / f"mind_{split}_news.parquet"
        if input_file.exists():
            output_file = silver_dir / f"mind_{split}_news_features.parquet"
            process_parquet_file(
                str(input_file), 
                str(output_file), 
                political_terms, 
                emotional_terms,
                args.sample
            )
    
    # Process Kaggle news
    kaggle_file = bronze_dir / "kaggle_news.parquet"
    if kaggle_file.exists():
        output_file = silver_dir / "kaggle_news_features.parquet"
        process_parquet_file(
            str(kaggle_file), 
            str(output_file), 
            political_terms, 
            emotional_terms,
            args.sample
        )
    
    # Process Dask dataset if it exists
    dask_dir = bronze_dir / "dask"
    if dask_dir.exists():
        output_dir = silver_dir / "dask"
        process_dask_dataset(
            str(dask_dir), 
            str(output_dir), 
            political_terms, 
            emotional_terms,
            args.workers
        )
    
    print("\nFeature engineering complete!")
    print(f"Processed features available in: {silver_dir}")
    print("\nNext step: Run model training with the silicon layer")

if __name__ == "__main__":
    main()