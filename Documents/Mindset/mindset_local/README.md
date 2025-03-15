# MINDSET: AI-Powered News Analytics Application

## Overview
MINDSET empowers readers with transparent, insightful metrics displayed alongside each news article:
- **Political Influence Level:** Visual gradient from neutral (green) to strong political bias (red).
- **Rhetoric Intensity Scale:** Visual gradient from informative, factual tone (blue) to highly emotionally charged language (red).
- **Information Depth Score:** Clearly labeled as either "Overview," "Analysis," or "In-depth".

## Architecture
MINDSET follows a medallion architecture with a Silicon Layer:
- **Raw Layer:** Local ingestion of Microsoft MINDLarge dataset, Kaggle News Category Dataset, and real-time article ingestion via NewsAPI.org
- **Bronze Layer:** Initial data cleaning, parsing, and structured formatting
- **Silver Layer:** Comprehensive feature engineering with Rust acceleration
- **Silicon Layer:** Advanced ML models, explainability, and drift detection
- **Gold Layer:** Production-level ML models serving via FastAPI

## Technology Stack
- **Backend:**
  - Python (FastAPI, scikit-learn, PyTorch, Hugging Face Transformers)
  - Rust via PyO3 for optimized performance-intensive computations
  - DuckDB for efficient local analytical query performance
  - Apache Arrow & Parquet for optimal data storage and manipulation
  - Dask for scalable parallel data processing locally

- **Frontend:**
  - React.js with Next.js
  - Tailwind CSS for styling

## Setup and Installation

### Prerequisites
- Python 3.8-3.11
- Node.js 16+ and npm
- Rust (if using Rust acceleration)
- Git LFS (for handling large model files)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/mindset.git
cd mindset
```

### Step 2: Set Up Environment
```bash
# Run the environment setup script
python mindset_local/scripts/setup_environment.py

# Create symbolic links to the MIND dataset
python mindset_local/scripts/setup_data_links.py

# Optional: Download Kaggle News Category Dataset
# You'll need to set up Kaggle API credentials first
python mindset_local/scripts/download_kaggle_dataset.py
```

### Step 3: Process Datasets
```bash
# Process raw data into structured formats (Bronze layer)
python mindset_local/scripts/process_datasets.py
```

### Step 4: Feature Engineering
```bash
# Generate features for machine learning (Silver layer)
python mindset_local/scripts/feature_engineering.py
```

### Step 5: Train Models (Silicon Layer)
```bash
# Train and evaluate models
python mindset_local/scripts/train_models.py
```

### Step 6: Run the Application
```bash
# Run both the backend API and frontend in one command
python mindset_local/scripts/run_application.py
```

Then open your browser and navigate to:
- Frontend: http://localhost:3000
- API: http://localhost:8000

## Individual Component Usage

### Running Only the Backend
```bash
python mindset_local/scripts/run_api.py --host 127.0.0.1 --port 8000
```

### Running Only the Frontend
```bash
python mindset_local/scripts/run_frontend.py --host localhost --port 3000 --api-url http://127.0.0.1:8000
```

### API Endpoints
- `GET /` - API info
- `GET /health` - Health check
- `GET /models` - Get model information
- `POST /analyze` - Analyze an article
- `POST /batch-analyze` - Analyze multiple articles
- `POST /explain` - Get explanation for article metrics
- `POST /detect-drift` - Detect data drift in a batch of articles

## Project Structure
```
mindset_local/
├── backend/            # FastAPI backend
├── frontend/           # React.js frontend
├── scripts/            # Utility scripts
├── silicon_layer/      # ML models and processing
│   ├── ensemble_model.py   # Stacked ensemble model implementation
│   ├── xai_wrapper.py      # Explainable AI integration
│   ├── drift_detector.py   # Data and concept drift detection
│   └── silicon_layer.py    # Integration layer
├── rust_modules/       # Rust acceleration modules
└── data/               # Data storage
    ├── raw/            # Raw data
    ├── bronze/         # Cleaned data
    ├── silver/         # Feature-engineered data
    └── gold/           # Model predictions
```

## Main Features

### Article Analysis
MINDSET provides three key metrics for each analyzed article:

1. **Political Influence Score**
   - Measures the level of political bias in the content
   - Visualized as a gradient from green (neutral) to red (strong bias)

2. **Rhetoric Intensity Score**
   - Measures the emotional tone of the language
   - Visualized as a gradient from blue (informative/factual) to red (emotionally charged)

3. **Information Depth Score**
   - Measures the depth and complexity of information
   - Categorized as "Overview," "Analysis," or "In-depth"

### Explainability
MINDSET provides explanations for its metrics using:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

### Drift Detection
The system monitors for both:
- Data drift: Changes in feature distributions over time
- Concept drift: Changes in the relationship between features and predictions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Missing Dependencies
If you see warnings about missing dependencies like:
```
WARNING - SHAP and/or LIME not available. Install with: pip install shap lime
```

Install the required explainability packages:
```bash
# Option 1: Install directly
pip install shap lime

# Option 2: Use the requirements file
pip install -r requirements-explainability.txt
```

### API Connection Issues
If the frontend shows "No response from server" errors:
1. Ensure the backend is running (check with `http://localhost:8000/health`)
2. Verify the frontend is using the correct API URL:
   - Set the environment variable: `REACT_APP_API_URL=http://localhost:8000`
   - Or update the default in `src/config.js`
3. Restart both frontend and backend servers

### Port Conflicts
If you see `Address already in use` errors:
```bash
# Find the process using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or start the server on a different port
python backend/app.py --port 8001
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
