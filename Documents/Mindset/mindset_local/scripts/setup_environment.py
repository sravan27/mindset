#!/usr/bin/env python3
"""
MINDSET Environment Setup Script
This script sets up all necessary components for running MINDSET locally.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8) or sys.version_info >= (3, 12):
        print("Error: Python 3.8-3.11 is required")
        sys.exit(1)
    print("✓ Python version check passed")

def create_directory_structure():
    """Create the directory structure for MINDSET."""
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "bronze",
        base_dir / "data" / "silver",
        base_dir / "data" / "gold",
        base_dir / "backend",
        base_dir / "frontend",
        base_dir / "silicon_layer",
        base_dir / "rust_modules",
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    
    print("✓ Directory structure created")

def install_python_dependencies():
    """Install Python dependencies."""
    base_dir = Path(__file__).parent.parent
    requirements_file = base_dir / "requirements.txt"
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("✓ Python dependencies installed")
    except subprocess.CalledProcessError:
        print("Error: Failed to install Python dependencies")
        sys.exit(1)

def setup_rust_environment():
    """Set up Rust environment for PyO3."""
    try:
        # Check if Rust is installed
        result = subprocess.run(
            ["rustc", "--version"], 
            capture_output=True, 
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("Rust not found. Please install Rust first using:")
            print("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
            sys.exit(1)
            
        # Install maturin for PyO3 integration
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "maturin"],
            check=True
        )
        
        # Initialize Rust project structure
        base_dir = Path(__file__).parent.parent
        rust_dir = base_dir / "rust_modules"
        
        os.chdir(rust_dir)
        subprocess.run(
            ["maturin", "init", "--name", "mindset_rust"],
            check=True
        )
        
        print("✓ Rust environment set up")
    except subprocess.CalledProcessError:
        print("Error: Failed to set up Rust environment")
        sys.exit(1)

def setup_newsapi():
    """Set up NewsAPI configuration."""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "api_config.py"
    
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write('"""\nAPI Configuration for MINDSET\n"""\n\n')
            f.write('# Get your API key from https://newsapi.org/\n')
            f.write('NEWSAPI_KEY = "your_api_key_here"\n\n')
            f.write('# Kaggle API credentials\n')
            f.write('KAGGLE_USERNAME = "your_kaggle_username"\n')
            f.write('KAGGLE_KEY = "your_kaggle_key"\n')
    
    print("✓ API configuration template created at", config_file)
    print("  Please update with your actual API keys before proceeding")

def setup_mind_dataset_links():
    """Set up links to MIND dataset for local usage."""
    base_dir = Path(__file__).parent.parent
    mind_script = base_dir / "scripts" / "setup_data_links.py"
    
    with open(mind_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
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
''')
    
    os.chmod(mind_script, 0o755)  # Make executable
    print("✓ MIND dataset link script created")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up MINDSET environment")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust setup")
    args = parser.parse_args()
    
    print("Setting up MINDSET environment...")
    
    check_python_version()
    create_directory_structure()
    install_python_dependencies()
    
    if not args.skip_rust:
        setup_rust_environment()
    
    setup_newsapi()
    setup_mind_dataset_links()
    
    print("\nMINDSET environment setup complete!")
    print("\nNext steps:")
    print("1. Update API keys in config/api_config.py")
    print("2. Run scripts/setup_data_links.py to set up MIND dataset links")
    print("3. Run scripts/download_kaggle_dataset.py to download the Kaggle News Category Dataset")

if __name__ == "__main__":
    main()