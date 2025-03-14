#!/usr/bin/env python3
"""
MINDSET Pipeline Runner
This script runs the complete MINDSET data pipeline locally.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import importlib.util
import shutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def create_data_directories():
    """Create the necessary data directories if they don't exist"""
    for layer in ["raw", "bronze", "silver", "gold", "newsapi", "silicon"]:
        dir_path = DATA_DIR / layer
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    # Create specific subdirectories
    (DATA_DIR / "raw" / "newsapi").mkdir(exist_ok=True)
    (DATA_DIR / "gold" / "models").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "gold" / "silicon").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "silicon" / "features").mkdir(parents=True, exist_ok=True)

def run_module(module_path, function_name="main"):
    """
    Dynamically import and run a module function
    
    Args:
        module_path: Path to the Python module
        function_name: Name of the function to run (default: main)
    """
    try:
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            logger.info(f"Running {module_path.name}:{function_name}...")
            return func()
        else:
            logger.error(f"Function {function_name} not found in {module_path}")
            return False
    except Exception as e:
        logger.error(f"Error running {module_path}: {str(e)}")
        return False

def run_pipeline(steps=None):
    """
    Run the MINDSET data pipeline
    
    Args:
        steps: List of specific steps to run (if None, run all)
    """
    # Define pipeline steps
    pipeline_steps = {
        "newsapi": {
            "module": BASE_DIR / "src" / "data" / "raw" / "newsapi_ingest.py",
            "description": "Ingest data from NewsAPI"
        },
        "bronze": {
            "module": BASE_DIR / "src" / "data" / "bronze" / "preprocess.py",
            "description": "Preprocess raw data into bronze layer"
        },
        "silver": {
            "module": BASE_DIR / "src" / "data" / "silver" / "feature_engineering.py",
            "description": "Process bronze data into silver layer with features"
        },
        "silicon": {
            "module": BASE_DIR / "src" / "ml" / "silicon_layer" / "integrate_metrics_engine.py", 
            "description": "Apply Silicon Layer advanced ML processing"
        },
        "gold": {
            "module": BASE_DIR / "src" / "ml" / "models" / "metrics_model.py",
            "description": "Train and save models"
        }
    }
    
    # Determine which steps to run
    if steps:
        steps_to_run = [s for s in steps if s in pipeline_steps]
        if not steps_to_run:
            logger.error(f"No valid steps specified. Available steps: {list(pipeline_steps.keys())}")
            return False
    else:
        steps_to_run = list(pipeline_steps.keys())
    
    logger.info(f"Starting MINDSET pipeline with steps: {steps_to_run}")
    start_time = time.time()
    
    # Run each step
    results = {}
    for step in steps_to_run:
        step_info = pipeline_steps[step]
        logger.info(f"==== Running step: {step} - {step_info['description']} ====")
        
        step_start = time.time()
        success = run_module(step_info["module"])
        step_duration = time.time() - step_start
        
        results[step] = {
            "success": success,
            "duration": step_duration
        }
        
        if success:
            logger.info(f"Step {step} completed successfully in {step_duration:.2f} seconds")
        else:
            logger.error(f"Step {step} failed after {step_duration:.2f} seconds")
            if step in ["bronze", "silver"] and "continue_on_error" not in sys.argv:
                logger.error("Critical step failed. Stopping pipeline.")
                break
    
    # Print summary
    total_duration = time.time() - start_time
    logger.info(f"==== Pipeline completed in {total_duration:.2f} seconds ====")
    for step, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        logger.info(f"  {step}: {status} ({result['duration']:.2f}s)")
    
    # Return True if all steps succeeded
    return all(result["success"] for result in results.values())

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MINDSET Pipeline Runner")
    parser.add_argument("steps", nargs="*", help="Specific pipeline steps to run")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline even if a step fails")
    
    args = parser.parse_args()
    
    # Create data directories
    create_data_directories()
    
    # Run the pipeline
    success = run_pipeline(args.steps)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()