"""
MINDSET Backend API
FastAPI application for serving news article metrics
"""
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Silicon Layer
from silicon_layer.silicon_layer import SiliconLayer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mindset_api')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GOLD_DATA_DIR = DATA_DIR / "gold"
SILICON_DATA_DIR = DATA_DIR / "silicon"

# Create application
app = FastAPI(
    title="MINDSET API",
    description="API for News Transparency Metrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Silicon Layer
silicon_layer = SiliconLayer(
    base_dir=SILICON_DATA_DIR,
    use_feature_store=True,
    use_drift_detection=True,
    use_ensemble_models=True,
    use_xai=True,
    metrics_engine='python'
)

# Pydantic models
class Metrics(BaseModel):
    political_influence: float = Field(..., description="Political influence level (0-10)")
    rhetoric_intensity: float = Field(..., description="Rhetoric intensity scale (0-10)")
    information_depth: float = Field(..., description="Information depth score (0-10)")

class Recommendation(BaseModel):
    news_id: str
    title: str
    score: float

class Article(BaseModel):
    news_id: str
    title: str
    abstract: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[str] = None
    metrics: Metrics
    recommendations: Optional[List[Recommendation]] = None

class ArticleInput(BaseModel):
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    source_name: Optional[str] = None
    category: Optional[str] = None

class ArticleResponse(BaseModel):
    news_id: str
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    source_name: Optional[str] = None
    category: Optional[str] = None
    metrics: Metrics

class MetricsSummary(BaseModel):
    count: int
    averages: Metrics

# Cache for articles
articles_cache = None
last_cache_update = None

def get_cached_articles() -> List[Dict]:
    """
    Get articles from cache or load from Gold layer
    
    Returns:
        List of article dictionaries
    """
    global articles_cache, last_cache_update
    
    # Check if cache needs to be updated
    current_time = datetime.now()
    if (
        articles_cache is None or 
        last_cache_update is None or 
        (current_time - last_cache_update).total_seconds() > 300  # 5 minutes
    ):
        articles_cache = load_articles()
        last_cache_update = current_time
    
    return articles_cache

def load_articles() -> List[Dict]:
    """
    Load articles from Gold layer
    
    Returns:
        List of article dictionaries
    """
    # Find the latest Gold file
    gold_files = list(GOLD_DATA_DIR.glob("*.parquet"))
    
    if not gold_files:
        # If no Gold files, check for sample data
        sample_path = BASE_DIR / "data" / "sample_articles.json"
        
        if sample_path.exists():
            logger.info(f"Loading sample articles from {sample_path}")
            with open(sample_path, 'r') as f:
                return json.load(f)
        
        logger.warning("No article data found")
        return []
    
    # Load the latest file
    latest_gold_file = max(gold_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading articles from {latest_gold_file}")
    
    import pandas as pd
    articles_df = pd.read_parquet(latest_gold_file)
    
    # Convert to list of dictionaries
    articles = articles_df.to_dict('records')
    
    return articles

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "MINDSET API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/api/articles", response_model=List[Article])
async def get_articles(
    limit: int = Query(10, description="Maximum number of articles to return"),
    offset: int = Query(0, description="Number of articles to skip"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    Get a list of news articles with transparency metrics
    """
    articles = get_cached_articles()
    
    # Apply filters
    if category:
        articles = [a for a in articles if a.get('category') == category]
    
    # Apply pagination
    paginated = articles[offset:offset+limit]
    
    return paginated

@app.get("/api/articles/{news_id}", response_model=Article)
async def get_article(news_id: str):
    """
    Get a specific article by ID
    """
    articles = get_cached_articles()
    
    # Find the article
    article = next((a for a in articles if a.get('news_id') == news_id), None)
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return article

@app.post("/api/articles/analyze", response_model=ArticleResponse)
async def analyze_article(article: ArticleInput):
    """
    Analyze an article and return transparency metrics
    """
    # Create article dictionary
    article_dict = {
        'news_id': f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'title': article.title,
        'content': article.content or "",
        'url': article.url,
        'source_name': article.source_name,
        'category': article.category or "general"
    }
    
    try:
        # Process through Silicon Layer
        result = silicon_layer.process([article_dict])
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=500, detail="Error processing article")
        
        # Extract the first result
        processed = result[0]
        
        # Create response
        response = {
            'news_id': processed['news_id'],
            'title': processed['title'],
            'content': processed.get('content', ''),
            'url': processed.get('url', ''),
            'source_name': processed.get('source_name', ''),
            'category': processed.get('category', 'general'),
            'metrics': processed.get('metrics', {})
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics", response_model=MetricsSummary)
async def get_metrics_summary():
    """
    Get summary of metrics across all articles
    """
    articles = get_cached_articles()
    
    # Filter articles with metrics
    articles_with_metrics = [a for a in articles if 'metrics' in a and a['metrics']]
    
    if not articles_with_metrics:
        return {
            'count': 0,
            'averages': {
                'political_influence': 0,
                'rhetoric_intensity': 0,
                'information_depth': 0
            }
        }
    
    # Calculate averages
    count = len(articles_with_metrics)
    political_influence = sum(a['metrics'].get('political_influence', 0) for a in articles_with_metrics) / count
    rhetoric_intensity = sum(a['metrics'].get('rhetoric_intensity', 0) for a in articles_with_metrics) / count
    information_depth = sum(a['metrics'].get('information_depth', 0) for a in articles_with_metrics) / count
    
    return {
        'count': count,
        'averages': {
            'political_influence': political_influence,
            'rhetoric_intensity': rhetoric_intensity,
            'information_depth': information_depth
        }
    }

@app.get("/api/categories")
async def get_categories():
    """
    Get list of available categories
    """
    articles = get_cached_articles()
    
    # Extract unique categories
    categories = list(set(a.get('category') for a in articles if 'category' in a and a['category']))
    
    return {
        'categories': categories
    }

@app.get("/api/explain/{news_id}")
async def explain_article(news_id: str):
    """
    Get explanation for article metrics
    """
    articles = get_cached_articles()
    
    # Find the article
    article = next((a for a in articles if a.get('news_id') == news_id), None)
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    try:
        # Generate explanation
        explanation = silicon_layer.explain_article(article)
        
        return {
            'news_id': news_id,
            'title': article.get('title', ''),
            'explanations': explanation
        }
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)