#!/usr/bin/env python
"""
Run the MINDSET FastAPI backend
"""
import os
import sys
from pathlib import Path
import logging
import argparse
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_api_runner')

def main():
    parser = argparse.ArgumentParser(description="Run MINDSET API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent
    
    # Set environment variable for the project root
    os.environ["MINDSET_ROOT"] = str(project_root)
    
    # Add the project root to the Python path
    sys.path.append(str(project_root))
    
    # Run the API server
    logger.info(f"Starting API server at {args.host}:{args.port}")
    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()