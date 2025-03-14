#!/usr/bin/env python
"""
Setup data links from existing MIND datasets to the local_deployment data structure
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_data_setup')

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent

def setup_data_links():
    """
    Create symbolic links or copy MIND datasets to the local_deployment data structure
    """
    logger.info("Setting up data links from existing MIND datasets")
    
    # Source directories
    source_dirs = {
        "MINDlarge_train": PROJECT_ROOT / "MINDlarge_train",
        "MINDlarge_dev": PROJECT_ROOT / "MINDlarge_dev",
        "MINDlarge_test": PROJECT_ROOT / "MINDlarge_test",
        "MINDsmall_train": PROJECT_ROOT / "MINDsmall_train",
        "MINDsmall_dev": PROJECT_ROOT / "MINDsmall_dev",
        "MINDsmall_test": PROJECT_ROOT / "MINDsmall_test"
    }
    
    # Target directory
    target_dir = LOCAL_DEPLOYMENT_ROOT / "data" / "raw" / "MIND"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which source directories exist
    for source_name, source_path in source_dirs.items():
        if source_path.exists():
            logger.info(f"Found source directory: {source_path}")
            
            # Create target subdirectory
            target_subdir = target_dir / source_name
            target_subdir.mkdir(exist_ok=True)
            
            # Link or copy files
            for file_path in source_path.glob("*"):
                target_file = target_subdir / file_path.name
                
                if not target_file.exists():
                    try:
                        # Try to create symbolic link first
                        logger.info(f"Creating symbolic link from {file_path} to {target_file}")
                        os.symlink(file_path, target_file)
                    except (OSError, NotImplementedError):
                        # Fall back to copying file
                        logger.info(f"Copying from {file_path} to {target_file}")
                        shutil.copy2(file_path, target_file)
                else:
                    logger.info(f"Target file already exists: {target_file}")
        else:
            logger.warning(f"Source directory not found: {source_path}")
    
    # Generate sample data if no datasets were found
    if not any(source_path.exists() for source_path in source_dirs.values()):
        logger.warning("No MIND datasets found. Creating sample data.")
        create_sample_data()

def create_sample_data():
    """
    Create sample data if no MIND datasets are available
    """
    logger.info("Creating sample data")
    
    # Create sample articles
    sample_articles = []
    
    # Source directory for sample data
    sample_dir = LOCAL_DEPLOYMENT_ROOT / "data" / "raw" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 sample articles
    for i in range(10):
        article = {
            "news_id": f"sample_{i}",
            "category": "news" if i % 3 == 0 else "politics" if i % 3 == 1 else "technology",
            "subcategory": "general",
            "title": f"Sample Article {i}: Understanding the Impact of Technology",
            "abstract": f"This is sample article {i} discussing technology impact on society and economy.",
            "url": f"https://example.com/sample/{i}",
            "title_entities": "[]",
            "abstract_entities": "[]"
        }
        sample_articles.append(article)
    
    # Save as TSV
    import pandas as pd
    sample_df = pd.DataFrame(sample_articles)
    
    # Create MINDsmall_train directory
    mind_dir = LOCAL_DEPLOYMENT_ROOT / "data" / "raw" / "MIND" / "MINDsmall_train"
    mind_dir.mkdir(parents=True, exist_ok=True)
    
    # Save news.tsv
    news_path = mind_dir / "news.tsv"
    sample_df.to_csv(news_path, sep='\t', index=False, header=False)
    
    logger.info(f"Created sample news file at {news_path}")
    
    # Create behaviors.tsv with empty content
    behaviors_path = mind_dir / "behaviors.tsv"
    with open(behaviors_path, 'w') as f:
        f.write("")
    
    logger.info(f"Created sample behaviors file at {behaviors_path}")

if __name__ == "__main__":
    setup_data_links()
    logger.info("Data setup complete. You can now run process_datasets.py")