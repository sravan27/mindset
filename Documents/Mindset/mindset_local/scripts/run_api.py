#!/usr/bin/env python3
"""
MINDSET API Runner
Runs the FastAPI backend for the MINDSET application.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.run_api')

def run_api(host="127.0.0.1", port=8000, reload=True):
    """
    Run the FastAPI app with Uvicorn.
    
    Args:
        host: Host to listen on
        port: Port to listen on
        reload: Whether to enable auto-reload
    """
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    backend_dir = base_dir / "backend"
    
    # Export environment variables
    os.environ["MINDSET_BASE_DIR"] = str(base_dir)
    
    logger.info(f"Starting API server at {host}:{port}")
    logger.info(f"Using base directory: {base_dir}")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Build the Uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]
    
    if reload:
        cmd.append("--reload")
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running API server: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the API."""
    parser = argparse.ArgumentParser(description="Run MINDSET API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    # Run the API
    run_api(host=args.host, port=args.port, reload=not args.no_reload)

if __name__ == "__main__":
    main()