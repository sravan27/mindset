#!/usr/bin/env python
"""
Set up MINDSET local environment
Creates directories, installs dependencies, and generates initial data
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
import shutil
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_setup')

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pandas
        import numpy
        import duckdb
        import pyarrow
        import fastapi
        logger.info("Core Python dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def create_directory_structure():
    """Create directory structure for MINDSET"""
    logger.info("Creating directory structure")
    
    # Directories to create
    directories = [
        "data/raw",
        "data/bronze",
        "data/silver",
        "data/silicon",
        "data/gold",
        "models",
        "frontend/public",
        "frontend/styles",
    ]
    
    # Create directories
    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def install_python_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies")
    
    requirements_path = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_path.exists():
        logger.error(f"Requirements file not found: {requirements_path}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True
        )
        logger.info("Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Python dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    logger.info("Installing frontend dependencies")
    
    frontend_dir = PROJECT_ROOT / "frontend"
    
    if not (frontend_dir / "package.json").exists():
        logger.error(f"package.json not found in {frontend_dir}")
        return False
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Run npm install
        subprocess.run(["npm", "install"], check=True)
        
        # Change back to project root
        os.chdir(PROJECT_ROOT)
        
        logger.info("Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing frontend dependencies: {e}")
        return False
    except FileNotFoundError:
        logger.error("npm command not found. Please install Node.js")
        return False

def create_sample_env_file():
    """Create sample .env file"""
    logger.info("Creating sample .env file")
    
    env_path = PROJECT_ROOT / ".env"
    sample_content = """# MINDSET Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000/api

# Data Paths
DATA_DIR=./data
MODELS_DIR=./models

# Features
ENABLE_SILICON_LAYER=true
USE_RUST_ENGINE=false

# Optional: NewsAPI Key for fetching live articles
# NEWSAPI_KEY=your_key_here
"""
    
    with open(env_path, 'w') as f:
        f.write(sample_content)
    
    logger.info(f"Created sample .env file: {env_path}")

def main():
    parser = argparse.ArgumentParser(description="Set up MINDSET local environment")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--frontend-only", action="store_true", help="Set up frontend only")
    parser.add_argument("--backend-only", action="store_true", help="Set up backend only")
    args = parser.parse_args()
    
    logger.info("Starting MINDSET local setup")
    
    # Create directory structure
    create_directory_structure()
    
    # Create sample .env file
    create_sample_env_file()
    
    # Install dependencies if not skipped
    if not args.skip_deps:
        if not args.frontend_only:
            install_python_dependencies()
        
        if not args.backend_only and shutil.which('npm'):
            install_frontend_dependencies()
    
    logger.info("MINDSET local setup complete")
    logger.info("\nNext steps:")
    logger.info("1. Edit the .env file if needed")
    logger.info("2. Run 'python scripts/download_datasets.py' to download datasets")
    logger.info("3. Run 'python scripts/process_datasets.py' to process the data")
    logger.info("4. Run 'python scripts/train_models.py' to train the models")
    logger.info("5. Run 'python scripts/run_api.py' to start the backend API")
    logger.info("6. In another terminal, run 'cd frontend && npm run dev' to start the frontend")

if __name__ == "__main__":
    main()