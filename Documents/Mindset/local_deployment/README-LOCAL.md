# MINDSET Local Development Guide

This guide provides step-by-step instructions for setting up and running MINDSET locally without containers or cloud infrastructure.

## Overview

MINDSET is a news analytics platform with a Silicon Layer that provides transparency metrics for news articles:
- **Political Influence Level** (0-10): Measures political bias in content
- **Rhetoric Intensity Scale** (0-10): Measures emotional and persuasive language
- **Information Depth Score** (0-10): Assesses content depth and substance

## Prerequisites

- Python 3.9+ with pip installed
- Node.js 16+ with npm installed
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sravan27/mindset.git
cd mindset/local_deployment
```

### 2. Run the Setup Script

```bash
python scripts/setup_local.py
```

This script will:
- Create the necessary directory structure
- Install Python dependencies
- Install frontend dependencies
- Create a sample `.env` file

### 3. Download and Process Datasets

```bash
# Download datasets
python scripts/download_datasets.py

# Process datasets
python scripts/process_datasets.py
```

The data will flow through the medallion architecture:
- Raw → Bronze → Silver → Silicon → Gold

### 4. Train ML Models

```bash
python scripts/train_models.py
```

This will train ensemble models for each transparency metric and integrate them with the Silicon Layer.

### 5. Start the Backend API

```bash
python scripts/run_api.py
```

The API will be available at http://localhost:8000.
- API documentation: http://localhost:8000/docs

### 6. Start the Frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000.

## Project Structure

```
mindset/local_deployment/
├── data/                 # Data storage
│   ├── raw/              # Raw data
│   ├── bronze/           # Cleaned data
│   ├── silver/           # Transformed data
│   ├── silicon/          # ML features
│   └── gold/             # Final processed data
├── models/               # Trained models
├── silicon_layer/        # Silicon Layer code
│   ├── feature_store.py  # Feature storage and versioning
│   ├── drift_detector.py # Drift detection
│   ├── ensemble_model.py # Ensemble learning
│   ├── xai_wrapper.py    # Explainable AI
│   └── silicon_layer.py  # Main module
├── backend/              # FastAPI backend
│   └── app.py            # Main API application
├── frontend/             # Next.js frontend
│   ├── components/       # React components
│   ├── pages/            # Next.js pages
│   └── styles/           # CSS styles
└── scripts/              # Utility scripts
    ├── setup_local.py    # Setup script
    ├── download_datasets.py  # Dataset downloader
    ├── process_datasets.py   # Data processing
    ├── train_models.py       # Model training
    └── run_api.py            # API runner
```

## API Endpoints

- `GET /api/articles` - Get list of articles with metrics
- `GET /api/articles/{news_id}` - Get a specific article
- `POST /api/articles/analyze` - Analyze a new article
- `GET /api/metrics` - Get metrics summary
- `GET /api/categories` - Get list of categories
- `GET /api/explain/{news_id}` - Get explanation for article metrics
- `GET /api/health` - Health check

## Frontend Pages

- `/` - Home page with article list and metrics summary
- `/article/[id]` - Article detail page with metrics and explanations

## Advanced Usage

### Customizing Data Processing

Edit `scripts/process_datasets.py` to modify data processing steps.

### Adding New Metrics

1. Add new metrics to `silicon_layer/silicon_layer.py` in the `_calculate_metrics_rule_based` method
2. Update the models in `scripts/train_models.py`
3. Update the API and frontend to display the new metrics

### Using Live News Data

1. Get an API key from [NewsAPI](https://newsapi.org/)
2. Add your key to the `.env` file: `NEWSAPI_KEY=your_key_here`
3. Run `python scripts/download_datasets.py` to fetch live news

## Troubleshooting

### API Connection Issues

If the frontend can't connect to the API:
1. Ensure the API is running on port 8000
2. Check that CORS is properly configured in `backend/app.py`
3. Verify the `NEXT_PUBLIC_API_URL` in the `.env` file

### Model Training Errors

If model training fails:
1. Check the error message in the logs
2. Ensure you have enough memory for training
3. Try reducing the `--synthetic-size` parameter

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test locally
4. Submit a pull request

## License

MIT