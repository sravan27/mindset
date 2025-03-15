#!/usr/bin/env python3
"""
MIND Dataset Link Setup Script
This script creates symbolic links to the existing MIND dataset for the MINDSET application.
"""

import os
import sys
from pathlib import Path
import argparse

def setup_links(source_base_dir, target_base_dir):
    """Create symbolic links from source MIND dataset to target data directory."""
    # Define dataset splits
    splits = ["train", "dev", "test"]
    
    for split in splits:
        source_dir = Path(source_base_dir) / f"MINDlarge_{split}"
        target_dir = Path(target_base_dir) / "data" / "raw" / f"mind_{split}"
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(exist_ok=True, parents=True)
        
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
            continue
        
        # Link files
        for file_name in ["behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"]:
            source_file = source_dir / file_name
            target_file = target_dir / file_name
            
            if not source_file.exists():
                print(f"Warning: Source file {source_file} does not exist. Skipping.")
                continue
                
            if target_file.exists() or target_file.is_symlink():
                target_file.unlink()
                
            # Create the symbolic link
            target_file.symlink_to(source_file.resolve())
            print(f"Created link: {target_file} -> {source_file}")
    
    print("Dataset links setup complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up MIND dataset links for MINDSET")
    parser.add_argument(
        "--source", 
        default="/Users/sravansridhar/Documents/Mindset",
        help="Base directory containing MINDlarge_train, MINDlarge_dev, MINDlarge_test directories"
    )
    parser.add_argument(
        "--target", 
        default="/Users/sravansridhar/Documents/Mindset/mindset_local",
        help="Target MINDSET directory"
    )
    
    args = parser.parse_args()
    setup_links(args.source, args.target)
