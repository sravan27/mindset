#!/usr/bin/env python
"""
Script to download datasets for MINDSET local development.
Downloads:
- MIND dataset
- Sample news articles via NewsAPI (if key is provided)
"""
import os
import shutil
import urllib.request
import zipfile
import json
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MIND_DATA_DIR = RAW_DATA_DIR / "MIND"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MIND_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_mind_dataset(dataset_size="small", force=False):
    """
    Download MIND dataset.
    Args:
        dataset_size: small or large
        force: If True, download even if files already exist
    """
    valid_sizes = ["small", "large"]
    if dataset_size not in valid_sizes:
        raise ValueError(f"dataset_size must be one of {valid_sizes}")
    
    # URLs for MIND dataset
    mind_url_prefix = "https://mind201910small.blob.core.windows.net/release"
    if dataset_size == "large":
        mind_url_prefix = "https://mind201910.blob.core.windows.net/release"
    
    download_sets = ["train", "dev", "test"]
    
    for subset in download_sets:
        target_dir = MIND_DATA_DIR / f"MIND{dataset_size}_{subset}"
        if target_dir.exists() and not force:
            print(f"MIND{dataset_size}_{subset} already exists, skipping download")
            continue
        
        # Create directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download zip file
        zip_path = RAW_DATA_DIR / f"MIND{dataset_size}_{subset}.zip"
        url = f"{mind_url_prefix}/MIND{dataset_size}_{subset}.zip"
        
        print(f"Downloading {url} to {zip_path}")
        try:
            # Only download if file doesn't exist or force is True
            if not zip_path.exists() or force:
                urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue
        
        # Extract zip file
        print(f"Extracting {zip_path} to {target_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Remove zip file after extraction
            zip_path.unlink()
            print(f"Successfully extracted {zip_path}")
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")

def download_newsapi_articles(api_key=None, query="technology", num_articles=100):
    """
    Download news articles from NewsAPI.
    Args:
        api_key: NewsAPI key (if None, will use NEWSAPI_KEY from .env)
        query: Search query
        num_articles: Number of articles to download
    """
    # If API key is not provided, try to get it from environment variables
    if api_key is None:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            print("NewsAPI key not provided. Skipping NewsAPI download.")
            return
    
    # Import here to avoid dependency if key isn't provided
    try:
        from newsapi import NewsApiClient
    except ImportError:
        print("newsapi-python is not installed. Run 'pip install newsapi-python'")
        return
    
    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=api_key)
    
    # Download articles
    print(f"Downloading {num_articles} news articles from NewsAPI with query '{query}'")
    
    # Calculate number of pages (100 articles per page max)
    page_size = min(100, num_articles)
    num_pages = (num_articles + page_size - 1) // page_size
    
    all_articles = []
    for page in range(1, num_pages + 1):
        try:
            response = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page=page,
                page_size=page_size
            )
            if response and 'articles' in response:
                all_articles.extend(response['articles'])
                print(f"Downloaded page {page}/{num_pages}, got {len(response['articles'])} articles")
        except Exception as e:
            print(f"Error downloading from NewsAPI: {e}")
    
    # Save articles to JSON file
    newsapi_dir = RAW_DATA_DIR / "newsapi"
    newsapi_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = newsapi_dir / f"{query}_articles.json"
    with open(output_file, 'w') as f:
        json.dump(all_articles[:num_articles], f)
    
    print(f"Downloaded {len(all_articles[:num_articles])} articles to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for MINDSET")
    parser.add_argument("--mind-size", choices=["small", "large"], default="small",
                        help="Size of MIND dataset to download")
    parser.add_argument("--force", action="store_true",
                        help="Force download even if files exist")
    parser.add_argument("--newsapi-query", default="technology",
                        help="Query for NewsAPI articles")
    parser.add_argument("--newsapi-count", type=int, default=100,
                        help="Number of articles to download from NewsAPI")
    args = parser.parse_args()
    
    print("Downloading datasets for MINDSET")
    print(f"Data directory: {DATA_DIR}")
    
    # Download MIND dataset
    download_mind_dataset(dataset_size=args.mind_size, force=args.force)
    
    # Download NewsAPI articles
    download_newsapi_articles(query=args.newsapi_query, num_articles=args.newsapi_count)
    
    print("Download complete!")

if __name__ == "__main__":
    main()