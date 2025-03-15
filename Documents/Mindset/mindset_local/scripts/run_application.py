#!/usr/bin/env python3
"""
MINDSET Application Runner
Runs the complete MINDSET application with backend and frontend.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import threading
import signal
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset.run_application')

# Global variables to track subprocesses
processes = []

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    logger.info("Shutting down MINDSET application...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            logger.info(f"Terminating process with PID {process.pid}")
            process.terminate()
    
    # Wait for processes to terminate
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process with PID {process.pid} did not terminate gracefully, killing it")
            process.kill()
    
    logger.info("All processes terminated")
    sys.exit(0)

def run_backend(api_host, api_port):
    """Run the backend API."""
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    api_script = base_dir / "scripts" / "run_api.py"
    
    logger.info(f"Starting backend API on {api_host}:{api_port}")
    
    # Run the API script
    cmd = [
        sys.executable, str(api_script),
        "--host", api_host,
        "--port", str(api_port)
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(process)
        
        # Log the output in a separate thread
        def log_output():
            for line in process.stdout:
                logger.info(f"[API] {line.strip()}")
        
        threading.Thread(target=log_output, daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error starting backend: {e}")
        return None

def run_frontend(frontend_host, frontend_port, api_url):
    """Run the frontend server."""
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    frontend_script = base_dir / "scripts" / "run_frontend.py"
    
    logger.info(f"Starting frontend on {frontend_host}:{frontend_port}")
    logger.info(f"Frontend will connect to API at {api_url}")
    
    # Run the frontend script
    cmd = [
        sys.executable, str(frontend_script),
        "--host", frontend_host,
        "--port", str(frontend_port),
        "--api-url", api_url
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(process)
        
        # Log the output in a separate thread
        def log_output():
            for line in process.stdout:
                logger.info(f"[Frontend] {line.strip()}")
        
        threading.Thread(target=log_output, daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error starting frontend: {e}")
        return None

def main():
    """Main function to run the MINDSET application."""
    parser = argparse.ArgumentParser(description="Run MINDSET Application")
    parser.add_argument("--api-host", default="127.0.0.1", help="Host for the API backend")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the API backend")
    parser.add_argument("--frontend-host", default="localhost", help="Host for the frontend")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Port for the frontend")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Construct API URL
    api_url = f"http://{args.api_host}:{args.api_port}"
    
    # Start backend
    backend_process = run_backend(args.api_host, args.api_port)
    if not backend_process:
        logger.error("Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Wait a moment for backend to start
    logger.info("Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    frontend_process = run_frontend(args.frontend_host, args.frontend_port, api_url)
    if not frontend_process:
        logger.error("Failed to start frontend. Stopping backend and exiting.")
        backend_process.terminate()
        sys.exit(1)
    
    logger.info(f"MINDSET application is running!")
    logger.info(f"API is available at: {api_url}")
    logger.info(f"Frontend is available at: http://{args.frontend_host}:{args.frontend_port}")
    logger.info("Press Ctrl+C to shut down the application")
    
    try:
        # Keep the main thread alive
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                logger.error(f"Backend process exited with code {backend_process.returncode}")
                signal_handler(None, None)
                break
            
            if frontend_process.poll() is not None:
                logger.error(f"Frontend process exited with code {frontend_process.returncode}")
                signal_handler(None, None)
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()