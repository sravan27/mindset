# MINDSET Local Development Setup

This guide explains how to set up and run MINDSET locally without cloud infrastructure or containers.

## Overview

MINDSET is a news analytics platform with a Silicon Layer that provides transparency metrics:
- **Political Influence Level** (0-10): Measures political bias in content
- **Rhetoric Intensity Scale** (0-10): Measures emotional and persuasive language
- **Information Depth Score** (0-10): Assesses content depth and substance

## Architecture

- **Data Storage**: DuckDB, Apache Arrow & Parquet
- **Processing**: Dask for parallel processing
- **ML Pipeline (Silicon Layer)**: Transformers, Ensemble models
- **API**: FastAPI
- **Frontend**: React/Next.js with TailwindCSS

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 16+
- Git
- Visual Studio Code (recommended)

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv mindset_venv

# Activate on Windows
mindset_venv\Scripts\activate

# Activate on macOS/Linux
source mindset_venv/bin/activate
```

### 2. Clone Repository and Install Dependencies

```bash
git clone https://github.com/sravan27/mindset.git
cd mindset

# Install Python dependencies
pip install -r local_deployment/requirements.txt

# Install frontend dependencies
cd local_deployment/frontend
npm install
cd ../..
```

### 3. Download and Process Datasets

```bash
# Download datasets
python local_deployment/scripts/download_datasets.py

# Process datasets through medallion architecture
python local_deployment/scripts/process_datasets.py
```

### 4. Train ML Models

```bash
# Train and evaluate models
python local_deployment/scripts/train_models.py
```

### 5. Run Backend API

```bash
# Run FastAPI backend
python local_deployment/scripts/run_api.py
```

### 6. Run Frontend (in a separate terminal)

```bash
cd local_deployment/frontend
npm run dev
```

### 7. Access MINDSET

Open your browser and navigate to:
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

## Project Structure

```
mindset/
├── local_deployment/          # Local deployment code
│   ├── data/                  # Data storage
│   │   ├── raw/               # Raw data
│   │   ├── bronze/            # Cleaned data
│   │   ├── silver/            # Transformed data
│   │   ├── silicon/           # ML features
│   │   └── gold/              # Final processed data
│   ├── models/                # Trained models
│   ├── frontend/              # Next.js frontend
│   ├── backend/               # FastAPI backend
│   ├── notebooks/             # Jupyter notebooks
│   ├── scripts/               # Utility scripts
│   └── silicon_layer/         # ML pipeline code
└── README.md                  # Project documentation
```

## Data Flow

1. Raw data is loaded from sources (MIND dataset, NewsAPI)
2. Bronze layer cleans and standardizes the data
3. Silver layer performs feature engineering
4. Silicon layer adds ML-derived metrics:
   - Political Influence Level
   - Rhetoric Intensity Scale
   - Information Depth Score
5. Gold layer prepares data for frontend consumption

## Development Guidelines

- Follow the medallion architecture for data processing
- Document all code with docstrings
- Write unit tests for critical components
- Use black and flake8 for code formatting
- Use type hints for better code readability