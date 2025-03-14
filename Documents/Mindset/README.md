# MINDSET: AI-Driven News Recommendation System with Transparency Metrics

MINDSET is a sophisticated AI-driven news recommendation system that empowers users with transparency metrics to understand the nature of the articles they consume. Beyond standard content-based and collaborative filtering, MINDSET provides three key transparency metrics for every news article:

1. **Political Influence Level**: Measures political bias/leaning on a 0-10 scale
2. **Rhetoric Intensity Scale**: Quantifies emotional language on a 0-10 scale
3. **Information Depth Score**: Categorizes content as Shallow, Moderate, or Deep with a 0-10 score

The system processes news from the Microsoft MINDLarge dataset along with Kaggle News datasets and NewsAPI.org to create a comprehensive news analysis and recommendation platform.

## Architecture

MINDSET implements a modern medallion architecture with a custom Silicon Layer:

- **Raw Layer**: Original source data from MINDLarge dataset and NewsAPI
- **Bronze Layer**: Validated and lightly processed data
- **Silver Layer**: Feature-engineered data ready for ML
- **Silicon Layer**: Advanced ML processing with ensemble learning, XAI, and drift detection
- **Gold Layer**: Production-ready recommendations with transparency metrics

## Key Components

- **Silicon Layer**: The core ML engine with:
  - Feature Store for efficient feature management and versioning
  - Ensemble Learning with multiple model types (voting, stacking)
  - Explainable AI (XAI) for model transparency with SHAP/LIME
  - Drift Detection for data distribution monitoring
  - Online Learning for real-time model adaptation
  - Integration with Rust Metrics Engine for high-performance processing

- **Rust Metrics Engine**: High-performance calculation of transparency metrics
  - Python integration with PyO3
  - Optimized text analysis algorithms
  - Python fallback implementation when Rust is unavailable
  - Seamless batch processing capabilities

- **Data Pipeline**:
  - Raw to Bronze: Data ingestion and cleaning
  - Bronze to Silver: Feature engineering and initial metrics
  - Silver to Silicon: Advanced ML processing
  - Silicon to Gold: Optimized serving data and recommendations

- **Cloud Deployment**:
  - Primary: Azure ML, Blob Storage, AKS, CI/CD via Azure DevOps
  - Backup: AWS SageMaker, S3, ECS/EKS, CI/CD via AWS CodePipeline
  - Both ready to deploy with included scripts

## Deployment Options

MINDSET can be deployed on both Azure and AWS:

- **Azure Deployment**: See [README-AZURE-DEPLOY.md](README-AZURE-DEPLOY.md)
- **AWS Deployment**: See [README-AWS-DEPLOY.md](README-AWS-DEPLOY.md)

## Local Development

To run MINDSET locally:

1. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Build the Rust metrics engine:
   ```bash
   cd src/rust/metrics_engine
   python build.py
   ```

3. Test the Silicon Layer:
   ```bash
   python src/ml/silicon_layer_test.py
   ```

4. Run the complete pipeline:
   ```bash
   python run_pipeline.py --all
   ```
   
   Or run specific pipeline stages:
   ```bash
   python run_pipeline.py bronze silver silicon gold
   ```

5. Run the local development environment:
   ```bash
   docker-compose -f infrastructure/aws/docker-compose.yml up
   ```

6. Access the API at http://localhost:8080

## Dataset

The system uses the Microsoft MINDLarge dataset which should be placed in these directories:
- `MINDlarge_train/`
- `MINDlarge_dev/`
- `MINDlarge_test/`

You can download the dataset from the [Microsoft MIND website](https://msnews.github.io/).

## License

This project is licensed under the MIT License - see the LICENSE file for details.