#!/usr/bin/env python3
"""
MINDSET Frontend Runner
Runs the Next.js frontend for the MINDSET application.
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
logger = logging.getLogger('mindset.run_frontend')

def run_frontend(host="localhost", port=3000, api_url=None):
    """
    Run the Next.js frontend.
    
    Args:
        host: Host to listen on
        port: Port to listen on
        api_url: URL of the API backend
    """
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    frontend_dir = base_dir / "frontend"
    
    # Export environment variables
    os.environ["NEXT_PUBLIC_API_URL"] = api_url or f"http://{host}:8000"
    
    logger.info(f"Starting frontend server at {host}:{port}")
    logger.info(f"Using API URL: {os.environ['NEXT_PUBLIC_API_URL']}")
    logger.info(f"Using base directory: {base_dir}")
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Check if frontend dependencies are installed
    if not (frontend_dir / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        try:
            subprocess.run(["npm", "install"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            sys.exit(1)
    
    # Build the npm command
    cmd = [
        "npm", "run", "dev",
        "--", 
        "-H", host,
        "-p", str(port)
    ]
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Frontend server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running frontend server: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the frontend."""
    parser = argparse.ArgumentParser(description="Run MINDSET Frontend")
    parser.add_argument("--host", default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=3000, help="Port to listen on")
    parser.add_argument("--api-url", help="URL of the API backend")
    
    args = parser.parse_args()
    
    # Run the frontend
    run_frontend(host=args.host, port=args.port, api_url=args.api_url)

if __name__ == "__main__":
    main()