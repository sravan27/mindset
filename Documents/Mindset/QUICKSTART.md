# MINDSET Quick Start Guide

This guide helps you quickly deploy and run the MINDSET news recommendation platform with transparency metrics.

## Option 1: AWS EC2 Deployment (Recommended)

This deployment creates a single EC2 instance with all MINDSET components, including the Silicon Layer.

### Prerequisites

1. AWS CLI installed and configured with your credentials
2. Python 3.10+ installed
3. A NewsAPI.org API key

### Deployment Steps

1. Run the EC2 deployment script:
   ```bash
   ./deploy-mindset-aws-ec2.sh
   ```

2. Follow the prompts to enter your NewsAPI.org API key.

3. Wait for deployment to complete (10-15 minutes).

4. Access MINDSET in your browser at the URL provided in the output.

5. SSH into the instance to monitor setup:
   ```bash
   ssh -i <key-file>.pem ubuntu@<public-ip>
   sudo tail -f /var/log/user-data.log
   ```

6. View logs and troubleshoot if needed:
   ```bash
   sudo journalctl -u mindset-api.service
   sudo journalctl -u mindset-frontend.service
   sudo journalctl -u mindset-pipeline.service
   ```

## Option 2: Manual Installation

### Prerequisites

1. Ubuntu 22.04 LTS system
2. Python 3.10+
3. Node.js and npm
4. Rust installed

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Mindset.git
   cd Mindset
   ```

2. Set up Python environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Build the Rust metrics engine:
   ```bash
   cd src/rust/metrics_engine
   python build.py
   cd ../../..
   ```

4. Install frontend dependencies:
   ```bash
   cd src/frontend
   npm install
   cd ../..
   ```

5. Copy the `.env.example` file to `.env` and configure settings:
   ```bash
   cp .env.example .env
   # Edit .env with your settings, including your NewsAPI key
   ```

6. Run the data pipeline:
   ```bash
   python run_pipeline.py
   ```

7. Start the backend API:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

8. In another terminal, start the frontend:
   ```bash
   cd src/frontend
   npm run dev
   ```

9. Access MINDSET in your browser at http://localhost:3000

## Key Components

- **Frontend**: React/Next.js application that displays news articles with transparency metrics
- **Backend API**: FastAPI server that provides articles and metrics
- **Silicon Layer**: Advanced ML processing for transparency metrics
- **Rust Metrics Engine**: High-performance calculation of metrics
- **Data Pipeline**: Processes data through the medallion architecture (Raw → Bronze → Silver → Silicon → Gold)

## Transparency Metrics

MINDSET provides three key transparency metrics:

1. **Political Influence Level** (0-10): Measures political bias/leaning
2. **Rhetoric Intensity Scale** (0-10): Quantifies emotional language 
3. **Information Depth Score** (0-10): Categorizes content as Shallow, Moderate, or Deep

## Further Documentation

For more detailed information, see:
- README.md - General overview
- README-AWS.md - Detailed AWS deployment options
- API documentation at http://<your-instance>/api/docs