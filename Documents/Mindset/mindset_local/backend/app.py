"""
MINDSET FastAPI Backend
Serves ML models and article metrics via a RESTful API.
"""

import os
import sys
import json
import logging
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import re
import urllib.parse
import xml.etree.ElementTree as ET
import feedparser
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import uvicorn

# Add parent directory to path for local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from silicon_layer.silicon_layer import SiliconLayer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"mindset_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('mindset.api')

# Initialize FastAPI app
app = FastAPI(
    title="MINDSET API",
    description="API for MINDSET news analytics",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],  # Allow React dev server and all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly allow methods
    allow_headers=["Content-Type", "Authorization", "Accept"],  # Explicitly allow headers
    expose_headers=["Content-Length"],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Define request and response models
class ArticleRequest(BaseModel):
    """Article request model."""
    title: str = Field(..., description="Article title")
    abstract: Optional[str] = Field(None, description="Article abstract")
    content: Optional[str] = Field(None, description="Article full content")
    source: Optional[str] = Field(None, description="Article source")
    published_date: Optional[str] = Field(None, description="Article publication date")
    category: Optional[str] = Field(None, description="Article category")
    article_id: Optional[str] = Field(None, description="Article ID")
    url: Optional[HttpUrl] = Field(None, description="URL of the article")
    image_url: Optional[HttpUrl] = Field(None, description="URL of the article image")

class ArticleResponse(BaseModel):
    """Article response model with full details."""
    article_id: str = Field(..., description="Article ID")
    title: str = Field(..., description="Article title")
    abstract: str = Field(..., description="Article abstract")
    content: Optional[str] = Field(None, description="Article full content")
    source: Optional[str] = Field(None, description="Article source")
    published_date: Optional[str] = Field(None, description="Article publication date")
    category: Optional[str] = Field(None, description="Article category")
    url: Optional[HttpUrl] = Field(None, description="URL of the article")
    image_url: Optional[HttpUrl] = Field(None, description="URL of the article image")
    political_influence: float = Field(..., description="Political influence score (0-1)")
    rhetoric_intensity: float = Field(..., description="Rhetoric intensity score (0-1)")
    information_depth: float = Field(..., description="Information depth score (0-1)")
    information_depth_category: str = Field(..., description="Information depth category (Overview, Analysis, In-depth)")

class ArticleMetrics(BaseModel):
    """Article metrics response model."""
    article_id: str = Field(..., description="Article ID")
    political_influence: float = Field(..., description="Political influence score (0-1)")
    rhetoric_intensity: float = Field(..., description="Rhetoric intensity score (0-1)")
    information_depth: float = Field(..., description="Information depth score (0-1)")
    information_depth_category: str = Field(..., description="Information depth category (Overview, Analysis, In-depth)")

class NewsApiQuery(BaseModel):
    """NewsAPI query parameters."""
    query: Optional[str] = Field(None, description="Keywords or phrases to search for")
    sources: Optional[str] = Field(None, description="Comma-separated string of identifiers for news sources or blogs")
    domains: Optional[str] = Field(None, description="Comma-separated string of domains to restrict the search to")
    from_date: Optional[str] = Field(None, description="A date in ISO 8601 format (e.g., 2023-12-01)")
    to_date: Optional[str] = Field(None, description="A date in ISO 8601 format (e.g., 2023-12-31)")
    language: Optional[str] = Field("en", description="The 2-letter ISO-639-1 code of the language to get headlines for")
    sort_by: Optional[str] = Field("publishedAt", description="The order to sort articles in")

class ExplainResponse(BaseModel):
    """Explanation response model."""
    article_id: str = Field(..., description="Article ID")
    political_influence: float = Field(..., description="Political influence score")
    rhetoric_intensity: float = Field(..., description="Rhetoric intensity score")
    information_depth: float = Field(..., description="Information depth score")
    explanation: Dict[str, Any] = Field(..., description="Explanation details")

class DriftResponse(BaseModel):
    """Drift detection response model."""
    drift_detected: bool = Field(..., description="Whether drift is detected")
    date: str = Field(..., description="Date of drift detection")
    summary: Dict[str, Any] = Field(..., description="Summary of drift detection")
    drifted_features: Optional[Dict[str, Any]] = Field(None, description="Details of drifted features")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Timestamp of error")

# Global variables
base_dir = Path(os.getenv("MINDSET_BASE_DIR", parent_dir))
silicon_layer = None
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "dummy_key")  # Get from environment or use a placeholder
NEWS_API_URL = "https://newsapi.org/v2/"
NEWS_CACHE = {}  # Simple in-memory cache for news articles
ARTICLES_CACHE = []  # Cache of analyzed articles

# Define RSS feeds for guaranteed free news sources as a fallback
FREE_NEWS_FEEDS = [
    # General news sources
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "https://www.huffpost.com/section/front-page/feed",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/us/rss",
    
    # Technology
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.wired.com/wired/index",
    
    # Science
    "https://www.sciencedaily.com/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    
    # Business
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.feedburner.com/entrepreneur/latest"
]

# Option to use a better data source than NewsAPI
USE_ALTERNATIVE_NEWS_SOURCE = True  # Set to False to use NewsAPI exclusively

def get_silicon_layer():
    """Get or initialize the Silicon Layer."""
    global silicon_layer
    
    if silicon_layer is None:
        # Initialize Silicon Layer
        silicon_layer = SiliconLayer(
            base_dir=str(base_dir),
            data_dir=str(base_dir / "data"),
            model_dir=str(base_dir / "data" / "gold" / "models"),
            explainer_dir=str(base_dir / "data" / "silicon_layer" / "explainers"),
            drift_dir=str(base_dir / "data" / "silicon_layer" / "drift"),
            random_state=42
        )
        
        # Load models
        model_paths = silicon_layer.load_models()
        
        if not model_paths:
            logger.error("No models loaded. API will return errors for predictions.")
        else:
            logger.info(f"Loaded {len(model_paths)} models")
            
            # Try to load explainers
            if silicon_layer.load_explainers():
                logger.info("Explainers loaded successfully")
            else:
                logger.warning("Explainers not loaded. Explanations will not be available.")
            
            # Try to load drift detector
            if silicon_layer.load_drift_detector():
                logger.info("Drift detector loaded successfully")
            else:
                logger.warning("Drift detector not loaded. Drift detection will not be available.")
    
    return silicon_layer

def preprocess_article(article: ArticleRequest) -> pd.DataFrame:
    """Preprocess article data for prediction."""
    # Create DataFrame with article data
    df = pd.DataFrame([{
        "title": article.title,
        "abstract": article.abstract or "",
        "content": article.content or "",
        "source": article.source or "api",
        "category": article.category or "",
        "published_date": article.published_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    # Extract features needed for prediction
    # For now, we'll just use the title and abstract
    # This is simplified; in a real system, you'd have more complex feature engineering
    
    # Add any additional features required by the model
    # For example, text length, word count, etc.
    df["title_length"] = df["title"].str.len()
    df["abstract_length"] = df["abstract"].str.len()
    df["content_length"] = df["content"].str.len()
    df["word_count_title"] = df["title"].str.split().str.len()
    df["word_count_abstract"] = df["abstract"].str.split().str.len()
    
    return df

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info."""
    return {
        "api": "MINDSET News Analytics API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/analyze", response_model=ArticleMetrics)
async def analyze_article(
    article: ArticleRequest,
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """
    Analyze an article and return metrics.
    
    Takes an article (title, abstract, content) and returns metrics:
    - Political influence
    - Rhetoric intensity
    - Information depth
    """
    try:
        # Generate a unique ID if not provided
        article_id = article.article_id or f"article_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Preprocess article
        df = preprocess_article(article)
        
        # Make predictions
        predictions = sl.predict(df)
        
        if len(predictions) == 0:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Models not available."
            )
        
        # Compile response
        response = {
            "article_id": article_id,
            "political_influence": float(predictions["political_influence"].iloc[0]),
            "rhetoric_intensity": float(predictions["rhetoric_intensity"].iloc[0]),
            "information_depth": float(predictions["information_depth"].iloc[0]),
            "information_depth_category": predictions["information_depth_category"].iloc[0]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing article: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing article: {str(e)}"
        )

@app.post("/batch-analyze", response_model=List[ArticleMetrics])
async def batch_analyze(
    articles: List[ArticleRequest],
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """Analyze multiple articles in a batch."""
    try:
        results = []
        
        # Process each article
        for i, article in enumerate(articles):
            # Generate a unique ID if not provided
            article_id = article.article_id or f"article_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
            
            # Preprocess article
            df = preprocess_article(article)
            
            # Make predictions
            predictions = sl.predict(df)
            
            if len(predictions) == 0:
                continue
            
            # Compile response
            result = {
                "article_id": article_id,
                "political_influence": float(predictions["political_influence"].iloc[0]),
                "rhetoric_intensity": float(predictions["rhetoric_intensity"].iloc[0]),
                "information_depth": float(predictions["information_depth"].iloc[0]),
                "information_depth_category": predictions["information_depth_category"].iloc[0]
            }
            
            results.append(result)
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="Batch prediction failed. Models not available."
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch analysis: {str(e)}"
        )

@app.post("/explain", response_model=ExplainResponse)
async def explain_article(
    article: ArticleRequest,
    explanation_type: str = Query("shap", description="Type of explanation (shap or lime)"),
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """Explain the predictions for an article."""
    try:
        # Check if explainers are available
        if sl.xai_wrapper is None:
            raise HTTPException(
                status_code=501,
                detail="Explainability not available. Explainers not loaded."
            )
        
        # Generate a unique ID if not provided
        article_id = article.article_id or f"article_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Preprocess article
        df = preprocess_article(article)
        
        # Make predictions
        predictions = sl.predict(df)
        
        if len(predictions) == 0:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Models not available."
            )
        
        # Generate explanation
        explanation = sl.explain_prediction(df, index=0, explanation_type=explanation_type)
        
        if not explanation:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate {explanation_type} explanation."
            )
        
        # Compile response
        response = {
            "article_id": article_id,
            "political_influence": float(predictions["political_influence"].iloc[0]),
            "rhetoric_intensity": float(predictions["rhetoric_intensity"].iloc[0]),
            "information_depth": float(predictions["information_depth"].iloc[0]),
            "explanation": explanation
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining article: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error explaining article: {str(e)}"
        )

@app.post("/detect-drift", response_model=DriftResponse)
async def detect_drift(
    articles: List[ArticleRequest],
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """Detect drift in a batch of articles."""
    try:
        # Check if drift detector is available
        if sl.drift_detector is None:
            raise HTTPException(
                status_code=501,
                detail="Drift detection not available. Drift detector not loaded."
            )
        
        # Need at least a few articles for meaningful drift detection
        if len(articles) < 10:
            raise HTTPException(
                status_code=400,
                detail="Need at least 10 articles for meaningful drift detection."
            )
        
        # Preprocess articles
        all_df = pd.concat([preprocess_article(article) for article in articles], ignore_index=True)
        
        # Detect drift
        drift_results = sl.detect_drift(all_df)
        
        return drift_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting drift: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting drift: {str(e)}"
        )

@app.get("/articles", response_model=List[ArticleResponse])
async def get_articles(
    limit: int = Query(10, ge=1, le=50, description="Number of articles to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """
    Get list of articles with their metrics.
    
    This endpoint returns a list of articles with their metrics.
    It first tries to fetch real news articles from NewsAPI, then falls back to cached articles,
    and finally to sample articles if needed.
    """
    try:
        # If we have cached articles, use them but also try to fetch new ones
        cached_available = bool(ARTICLES_CACHE)
        
        # Only fetch new articles if the cache is small or we're at the beginning of the list
        # This avoids constantly fetching new articles with every request
        should_fetch_new = (len(ARTICLES_CACHE) < 20 or offset == 0)
        
        # Try to fetch real articles from NewsAPI
        if should_fetch_new:
            try:
                # Create a basic query for news
                news_query = NewsApiQuery(
                    language="en",
                    sort_by="publishedAt",
                    from_date=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
                )
                
                # Fetch news through our existing endpoint
                fresh_articles = await fetch_news(news_query, BackgroundTasks(), sl)
                
                if fresh_articles:
                    logger.info(f"Successfully fetched {len(fresh_articles)} fresh articles")
                    
                    # Check for duplicates before adding to cache
                    existing_ids = set(a.article_id for a in ARTICLES_CACHE)
                    existing_titles = set(a.title for a in ARTICLES_CACHE)
                    
                    # Filter out duplicates
                    unique_articles = [
                        article for article in fresh_articles 
                        if article.article_id not in existing_ids and article.title not in existing_titles
                    ]
                    
                    # Add unique articles to the beginning of cache
                    if unique_articles:
                        # Update the articles cache (global variable)
                        # Insert at beginning by creating a new list
                        cached_copy = list(ARTICLES_CACHE)  # Copy the current cache
                        ARTICLES_CACHE.clear()  # Clear the list
                        ARTICLES_CACHE.extend(unique_articles)  # Add new articles first
                        ARTICLES_CACHE.extend(cached_copy)  # Then add existing articles
                        logger.info(f"Added {len(unique_articles)} unique articles to cache")
                        
                        # Trim cache if it gets too large
                        if len(ARTICLES_CACHE) > 100:
                            del ARTICLES_CACHE[100:]  # Remove excess items
            
            except Exception as fetch_err:
                logger.warning(f"Error fetching fresh articles, will use cache: {str(fetch_err)}")
        
        # If we have cached articles now, return them
        if ARTICLES_CACHE:
            # Ensure we have valid pagination
            max_offset = max(0, len(ARTICLES_CACHE) - 1)
            safe_offset = min(offset, max_offset)
            
            # Slice according to pagination params
            paginated = ARTICLES_CACHE[safe_offset:safe_offset+limit]
            
            # If we got articles from cache, return them
            if paginated:
                return paginated
        
        # If cache is empty or pagination is beyond cache size, generate sample articles
        logger.warning("No articles in cache or pagination beyond cache size. Generating samples.")
        sample_articles = generate_sample_articles(sl, limit)
        
        # Add to cache if it's empty
        if not ARTICLES_CACHE:
            ARTICLES_CACHE.extend(sample_articles)
        
        return sample_articles
    
    except Exception as e:
        logger.error(f"Error fetching articles: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching articles: {str(e)}"
        )

@app.get("/article/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: str,
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """
    Get a specific article by ID with its metrics.
    
    This endpoint returns a specific article with its metrics.
    It looks for the article in the cache first, then generates a sample if not found.
    If the article has a URL, it attempts to fetch the full content.
    """
    try:
        # Look for article in cache
        cached_article = None
        for article in ARTICLES_CACHE:
            if article.article_id == article_id:
                cached_article = article
                break
        
        if cached_article:
            # If article exists but has no content and has a URL, try to fetch it
            if (not cached_article.content or len(cached_article.content) < 300) and cached_article.url:
                try:
                    logger.info(f"Attempting to fetch full content for article {article_id}")
                    full_content = fetch_article_content(str(cached_article.url))
                    if full_content:
                        # Create a new article response with the fetched content
                        cached_article = ArticleResponse(
                            article_id=cached_article.article_id,
                            title=cached_article.title,
                            abstract=cached_article.abstract,
                            content=full_content,  # Use the newly fetched content
                            source=cached_article.source,
                            published_date=cached_article.published_date,
                            category=cached_article.category,
                            url=cached_article.url,
                            image_url=cached_article.image_url,
                            political_influence=cached_article.political_influence,
                            rhetoric_intensity=cached_article.rhetoric_intensity,
                            information_depth=cached_article.information_depth,
                            information_depth_category=cached_article.information_depth_category
                        )
                        
                        # Update the cache with the enhanced article
                        for i, article in enumerate(ARTICLES_CACHE):
                            if article.article_id == article_id:
                                ARTICLES_CACHE[i] = cached_article
                                break
                                
                        logger.info(f"Updated article {article_id} with fetched content")
                except Exception as fetch_err:
                    logger.warning(f"Failed to fetch full content for article {article_id}: {str(fetch_err)}")
            
            return cached_article
        
        # If not found, generate a sample article with this ID
        sample_article = generate_sample_article(sl, article_id)
        
        # Try to fetch content if URL is available
        if sample_article.url:
            try:
                full_content = fetch_article_content(str(sample_article.url))
                if full_content:
                    # Update with fetched content
                    sample_article = ArticleResponse(
                        article_id=sample_article.article_id,
                        title=sample_article.title,
                        abstract=sample_article.abstract,
                        content=full_content,  # Use the fetched content
                        source=sample_article.source,
                        published_date=sample_article.published_date,
                        category=sample_article.category,
                        url=sample_article.url,
                        image_url=sample_article.image_url,
                        political_influence=sample_article.political_influence,
                        rhetoric_intensity=sample_article.rhetoric_intensity,
                        information_depth=sample_article.information_depth,
                        information_depth_category=sample_article.information_depth_category
                    )
            except Exception as fetch_err:
                logger.warning(f"Failed to fetch content for new article {article_id}: {str(fetch_err)}")
        
        # Add to cache
        ARTICLES_CACHE.append(sample_article)
        
        return sample_article
    
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching article {article_id}: {str(e)}"
        )

@app.post("/fetch-news", response_model=List[ArticleResponse])
async def fetch_news(
    query: NewsApiQuery,
    background_tasks: BackgroundTasks,
    sl: SiliconLayer = Depends(get_silicon_layer)
):
    """
    Fetch news articles from NewsAPI and analyze them.
    
    This endpoint fetches news articles from NewsAPI, analyzes them,
    and returns them with metrics.
    """
    try:
        # Check for cached results
        cache_key = f"{query.query}_{query.sources}_{query.domains}_{query.from_date}_{query.to_date}_{query.language}_{query.sort_by}"
        if cache_key in NEWS_CACHE:
            return NEWS_CACHE[cache_key]
        
        # Prepare API request parameters
        params = {
            "apiKey": NEWS_API_KEY,
            "language": query.language,
            "sortBy": query.sort_by,
        }
        
        # Add optional parameters if provided
        if query.query:
            params["q"] = query.query
        if query.sources:
            params["sources"] = query.sources
        if query.domains:
            params["domains"] = query.domains
        if query.from_date:
            params["from"] = query.from_date
        if query.to_date:
            params["to"] = query.to_date
        
        # Try to fetch real articles using the best available method
        articles = []
        
        # First check if we should use the alternative source (RSS feeds)
        if USE_ALTERNATIVE_NEWS_SOURCE:
            try:
                logger.info("Fetching news from RSS feeds (reliable source)")
                rss_articles = fetch_articles_from_rss(query)
                
                if rss_articles:
                    articles = rss_articles
                    logger.info(f"Successfully fetched {len(articles)} articles from RSS feeds")
            except Exception as rss_err:
                logger.error(f"Failed to fetch from RSS feeds: {str(rss_err)}", exc_info=True)
                
        # If alternative source didn't yield results or is disabled, try NewsAPI
        if not articles and NEWS_API_KEY != "dummy_key":
            try:
                logger.info("Attempting to fetch from NewsAPI")
                # First try the "everything" endpoint which provides more comprehensive results
                response = requests.get(f"{NEWS_API_URL}everything", params=params, timeout=10)
                if response.status_code == 200:
                    api_response = response.json()
                    if "articles" in api_response and len(api_response["articles"]) > 0:
                        articles = api_response["articles"]
                        logger.info(f"Successfully fetched {len(articles)} articles from NewsAPI 'everything' endpoint")
                
                # If no articles found or error, try the "top-headlines" endpoint
                if not articles:
                    # Prepare params for top-headlines (has different required parameters)
                    headline_params = {
                        "apiKey": NEWS_API_KEY,
                        "language": query.language,
                        "country": "us"  # Default to US news
                    }
                    if query.sources:
                        headline_params["sources"] = query.sources
                    if query.query:
                        headline_params["q"] = query.query
                        
                    response = requests.get(f"{NEWS_API_URL}top-headlines", params=headline_params, timeout=10)
                    if response.status_code == 200:
                        api_response = response.json()
                        if "articles" in api_response and len(api_response["articles"]) > 0:
                            articles = api_response["articles"]
                            logger.info(f"Successfully fetched {len(articles)} articles from NewsAPI 'top-headlines' endpoint")
                
                # Check for common API errors
                if response.status_code == 401:
                    logger.error("NewsAPI unauthorized - invalid API key")
                elif response.status_code == 429:
                    logger.error("NewsAPI rate limit exceeded")
                elif response.status_code != 200:
                    logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                    
            except Exception as api_err:
                logger.error(f"Failed to fetch from NewsAPI: {str(api_err)}", exc_info=True)
        
        # If neither source yielded results, use sample data as a last resort
        if not articles:
            articles = generate_sample_news_articles(10)
            logger.warning("All news sources failed. Using sample news articles as fallback")
        
        # Process and analyze each article
        analyzed_articles = []
        for i, article in enumerate(articles):
            # Create article request with unique ID
            article_id = f"news_{i}_{int(datetime.now().timestamp())}"
            article_request = ArticleRequest(
                title=article["title"],
                abstract=article.get("description", ""),
                content=article.get("content", ""),
                source=article.get("source", {}).get("name", ""),
                published_date=article.get("publishedAt", ""),
                category=article.get("category", "News"),
                article_id=article_id,
                url=article.get("url", None),
                image_url=article.get("urlToImage", None)
            )
            
            # Ensure we have good content, either from the API or by fetching
            if article.get("content") and len(article.get("content", "")) > 500:
                # NewsAPI already gave us good content, we can use it directly
                logger.info(f"Article {article_id} already has content from NewsAPI ({len(article.get('content'))} chars)")
            elif article_request.url:
                # Only try to fetch full content if the URL seems valid
                url = str(article_request.url)
                
                # Analyze the URL to check if it's likely to be accessible
                parsed_url = urllib.parse.urlparse(url)
                
                # List of domain patterns that usually have paywalls or blocks
                paywall_patterns = [
                    'nytimes.com', 'wsj.com', 'washingtonpost.com', 'bloomberg.com',
                    'ft.com', 'economist.com', 'newyorker.com', 'medium.com'
                ]
                
                # Check if URL is likely paywalled
                is_likely_paywalled = any(pattern in parsed_url.netloc.lower() for pattern in paywall_patterns)
                
                if not is_likely_paywalled:
                    try:
                        # Still try to fetch for non-paywalled sites
                        full_content = fetch_article_content(url)
                        if full_content and len(full_content) > 300:
                            article_request.content = full_content
                            logger.info(f"Successfully fetched content for article {article_id} ({len(full_content)} chars)")
                        else:
                            # If we don't get enough content, fallback to using description + title as content
                            fallback_content = f"{article.get('title', '')}\n\n{article.get('description', '')}"
                            article_request.content = fallback_content
                            logger.info(f"Using fallback content for article {article_id} - fetch failed or returned too little")
                    except Exception as fetch_err:
                        # If fetch fails, use the description as content
                        fallback_content = f"{article.get('title', '')}\n\n{article.get('description', '')}"
                        article_request.content = fallback_content
                        logger.warning(f"Failed to fetch content for article {article_id}: {str(fetch_err)} - Using fallback content")
                else:
                    # For likely paywalled sites, don't try to fetch, just use what we have
                    fallback_content = f"{article.get('title', '')}\n\n{article.get('description', '')}"
                    article_request.content = fallback_content
                    logger.info(f"Article {article_id} likely paywalled ({parsed_url.netloc}) - Using title/description as content")
            else:
                # No URL, ensure we have some content (use description at minimum)
                if not article_request.content or len(article_request.content) < 50:
                    fallback_content = f"{article.get('title', '')}\n\n{article.get('description', '')}"
                    article_request.content = fallback_content
                    logger.info(f"No URL for article {article_id} - Using title/description as content")
            
            # Analyze article
            df = preprocess_article(article_request)
            predictions = sl.predict(df)
            
            if len(predictions) == 0:
                # Use random metrics if prediction fails
                analyzed_article = ArticleResponse(
                    article_id=article_id,
                    title=article_request.title,
                    abstract=article_request.abstract or "",
                    content=article_request.content,
                    source=article_request.source,
                    published_date=article_request.published_date,
                    category=article_request.category,
                    url=article_request.url,
                    image_url=article_request.image_url,
                    political_influence=random.uniform(0.2, 0.8),
                    rhetoric_intensity=random.uniform(0.2, 0.8),
                    information_depth=random.uniform(0.2, 0.8),
                    information_depth_category=random.choice(["Overview", "Analysis", "In-depth"])
                )
            else:
                # Create response object with model predictions
                analyzed_article = ArticleResponse(
                    article_id=article_id,
                    title=article_request.title,
                    abstract=article_request.abstract or "",
                    content=article_request.content,
                    source=article_request.source,
                    published_date=article_request.published_date,
                    category=article_request.category,
                    url=article_request.url,
                    image_url=article_request.image_url,
                    political_influence=float(predictions["political_influence"].iloc[0]),
                    rhetoric_intensity=float(predictions["rhetoric_intensity"].iloc[0]),
                    information_depth=float(predictions["information_depth"].iloc[0]),
                    information_depth_category=predictions["information_depth_category"].iloc[0]
                )
            
            analyzed_articles.append(analyzed_article)
        
        # Store in cache
        NEWS_CACHE[cache_key] = analyzed_articles
        
        # Add to global articles cache
        background_tasks.add_task(update_articles_cache, analyzed_articles)
        
        return analyzed_articles
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}", exc_info=True)
        
        # For demo, return sample articles if API fails
        sample_articles = generate_sample_articles(sl, 10)
        return sample_articles

@app.get("/fetch-article")
async def fetch_article(url: str = Query(..., description="The URL of the article to fetch")):
    """
    Fetch an article from a URL.
    
    This endpoint fetches an article from a URL and extracts its content.
    It is used by the frontend to fetch article content for analysis.
    """
    try:
        logger.info(f"Fetching article from URL: {url}")
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL scheme: {url}"
            )
            
        # Extract domain
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if the domain is likely to be paywalled or blocked
        paywall_patterns = [
            'nytimes.com', 'wsj.com', 'washingtonpost.com', 'bloomberg.com',
            'ft.com', 'economist.com', 'newyorker.com', 'medium.com'
        ]
        
        is_paywalled = any(pattern in domain for pattern in paywall_patterns)
        if is_paywalled:
            logger.warning(f"URL {url} is likely paywalled or blocked for scraping")
            
        # Initialize variables
        title = "Untitled Article"
        image_url = None
        source = domain
        content = None
        description = ""
        
        # First try to make a HEAD request to validate the URL 
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            head_response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            
            # Check if URL is valid and accessible
            if head_response.status_code != 200:
                logger.warning(f"URL {url} returned status code {head_response.status_code}")
                
                # If the URL is not directly accessible but not a 404, we'll still try to fetch content
                if head_response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"URL not found: {url} (404 response)"
                    )
        except requests.RequestException as head_err:
            logger.warning(f"Error validating URL {url}: {str(head_err)}")
            # We'll still try to fetch content even if the HEAD request fails
        
        # Try multiple strategies to get metadata and content
        strategies = [
            "regular_fetch",  # Regular content fetching
            "mobile_user_agent",  # Try with mobile user agent
            "no_javascript",  # Try with javascript disabled
        ]
        
        for strategy in strategies:
            try:
                logger.info(f"Trying {strategy} for URL: {url}")
                
                # Set up strategy-specific headers
                fetch_headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
                
                if strategy == "regular_fetch":
                    fetch_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                elif strategy == "mobile_user_agent":
                    fetch_headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
                elif strategy == "no_javascript":
                    fetch_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    fetch_headers['Cookie'] = 'javascript=false'
                
                # Try to get content with this strategy
                if not content:
                    logger.info(f"Attempting content fetch with {strategy}")
                    content = fetch_article_content(url)
                    
                # If we got content, try to extract metadata
                if not title or not image_url:
                    try:
                        # Get a small chunk for metadata
                        response = requests.get(url, headers=fetch_headers, timeout=5, stream=True)
                        chunk = next(response.iter_content(chunk_size=10000))
                        chunk_text = chunk.decode('utf-8', errors='ignore')
                        response.close()
                        
                        # Extract title
                        if not title or title == "Untitled Article":
                            # Try og:title first
                            og_title = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\'](.*?)["\']', chunk_text, re.IGNORECASE)
                            if og_title:
                                title = og_title.group(1).strip()
                            else:
                                # Try regular title tag
                                title_match = re.search(r'<title[^>]*>(.*?)</title>', chunk_text, re.IGNORECASE)
                                if title_match:
                                    title = title_match.group(1).strip()
                        
                        # Extract description
                        if not description:
                            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', chunk_text, re.IGNORECASE)
                            if desc_match:
                                description = desc_match.group(1).strip()
                            else:
                                og_desc = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\'](.*?)["\']', chunk_text, re.IGNORECASE)
                                if og_desc:
                                    description = og_desc.group(1).strip()
                        
                        # Extract image
                        if not image_url:
                            image_match = re.search(r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\'](.*?)["\']', chunk_text, re.IGNORECASE)
                            if image_match:
                                image_url = image_match.group(1).strip()
                                
                    except Exception as meta_err:
                        logger.warning(f"Error extracting metadata with {strategy}: {str(meta_err)}")
                
                # If we have both content and metadata, we can stop trying strategies
                if content and title != "Untitled Article" and image_url:
                    break
                    
            except Exception as strategy_err:
                logger.warning(f"Strategy {strategy} failed for {url}: {str(strategy_err)}")
        
        # If we still don't have content, use description as a fallback
        if not content and description:
            content = f"{title}\n\n{description}"
            logger.warning(f"Using fallback content (description) for {url}")
        elif not content:
            raise HTTPException(
                status_code=404,
                detail=f"Could not extract content from: {url} after trying multiple methods"
            )
        
        # Return the article data
        return {
            "url": url,
            "title": title,
            "content": content,
            "description": description,
            "source": source,
            "image_url": image_url
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching article from {url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching article: {str(e)}"
        )

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/models", response_model=Dict[str, Any])
async def get_models(sl: SiliconLayer = Depends(get_silicon_layer)):
    """Get information about the loaded models."""
    try:
        model_info = {}
        
        # Check if models are loaded
        if sl.ensemble_trainer is None or not any(model for model in sl.ensemble_trainer.models.values()):
            return {"status": "No models loaded"}
        
        # Get model information
        for name, model in sl.ensemble_trainer.models.items():
            if model is None or not hasattr(model, 'is_fitted') or not model.is_fitted:
                model_info[name] = {"status": "not fitted"}
                continue
            
            # Add basic model info
            model_info[name] = {
                "status": "fitted",
                "feature_count": len(model.feature_cols) if model.feature_cols else 0
            }
            
            # Add top features if available
            if hasattr(model, 'get_feature_importance'):
                try:
                    feature_importance = model.get_feature_importance()
                    model_info[name]["top_features"] = dict(list(feature_importance.items())[:5])
                except Exception:
                    pass
        
        return {
            "models": model_info,
            "explainers_available": sl.xai_wrapper is not None,
            "drift_detector_available": sl.drift_detector is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )

# Article fetching helper function
def fetch_article_content(url: str) -> Optional[str]:
    """
    Fetch article content from a URL.
    
    This function attempts to fetch and extract the main content from an article URL.
    It includes robust error handling and multiple extraction methods with fallbacks.
    
    Args:
        url: The URL of the article to fetch
        
    Returns:
        Optional[str]: The extracted article content or None if fetching fails
    """
    logger.info(f"Attempting to fetch article content from {url}")
    
    # Skip non-http URLs
    if not url.startswith(('http://', 'https://')):
        logger.warning(f"Invalid URL scheme: {url}")
        return None
        
    # Avoid example domains and placeholder URLs
    if "example.com" in url or "example.org" in url:
        logger.warning(f"Skipping example domain URL: {url}")
        return None
    
    # Common headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }
    
    content = None
    extraction_attempts = 0
    max_attempts = 3
    
    # Try multiple attempts with different approaches
    while extraction_attempts < max_attempts and not content:
        extraction_attempts += 1
        
        try:
            logger.info(f"Article fetch attempt {extraction_attempts} for {url}")
            
            # Attempt to fetch the content with a timeout (increasing with each attempt)
            timeout = 5 + (extraction_attempts * 5)  # 10, 15, 20 seconds
            
            # Disable redirects on later attempts to avoid redirect loops
            allow_redirects = extraction_attempts < 2
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout,
                allow_redirects=allow_redirects,
                stream=(extraction_attempts > 1)  # Stream on later attempts to handle large pages
            )
            
            # Check response
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} response from {url}")
                if extraction_attempts == max_attempts:
                    return None
                continue  # Try again with different settings
                
            # Get content, handling streaming if needed
            if extraction_attempts > 1 and response.headers.get('content-length') and int(response.headers['content-length']) > 1000000:
                # For large pages, just read the beginning where the article content likely is
                html_content = b"".join(response.iter_content(chunk_size=None, decode_unicode=False) for _ in range(20))
                html_content = html_content.decode('utf-8', errors='ignore')
            else:
                html_content = response.text
                
            # Alternative content extraction strategies based on attempt number
            if extraction_attempts == 1:
                # First attempt: Try pattern-based extraction
                content = extract_content_by_patterns(html_content)
            elif extraction_attempts == 2:
                # Second attempt: Try DOM-based extraction
                content = extract_content_by_structure(html_content, url)
            else:
                # Final attempt: Most aggressive extraction
                content = extract_content_aggressive(html_content)
                
            # If we found content, clean and validate it
            if content:
                # Basic HTML entity decoding
                content = html_entity_decode(content)
                
                # Validate length
                if len(content) < 300:
                    logger.warning(f"Extracted content too short ({len(content)} chars), might be incomplete")
                    content = None  # Reset to try next method
                else:
                    logger.info(f"Successfully extracted {len(content)} chars of content from {url}")
                    break  # Success!
        
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout while fetching article from {url} (attempt {extraction_attempts})")
            # Try again with a different approach
        except requests.exceptions.TooManyRedirects:
            logger.warning(f"Too many redirects while fetching article from {url}")
            return None  # No point in retrying with redirects
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error while fetching article from {url}")
            # Try again with a different approach
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error while fetching article from {url}: {e}")
            return None  # HTTP errors unlikely to be resolved with retries
        except Exception as e:
            logger.warning(f"Unexpected error while fetching article from {url}: {e}")
            # Try again with a different approach
    
    # If all attempts failed
    if not content:
        logger.warning(f"All extraction methods failed for {url}")
        return None
        
    return content

def extract_content_by_patterns(html_content: str) -> Optional[str]:
    """Extract content using regex patterns targeting common article containers."""
    
    # Extract article content using various heuristics
    # First try to find common article content containers
    article_patterns = [
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*story[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*id="[^"]*article[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*post-content[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.*?)</div>'
    ]
    
    # Try each pattern
    for pattern in article_patterns:
        try:
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Take the longest match as it's likely the main content
                extracted_content = max(matches, key=len)
                
                # Extract paragraphs from this container
                paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', extracted_content, re.DOTALL)
                
                # Filter very short paragraphs
                paragraphs = [p for p in paragraphs if len(p.strip()) > 80]
                
                if paragraphs:
                    # Clean HTML tags from paragraphs
                    cleaned_paragraphs = [re.sub(r'<[^>]*>', ' ', p) for p in paragraphs]
                    # Join with proper paragraph breaks
                    joined_content = "\n\n".join(cleaned_paragraphs)
                    # Clean whitespace
                    joined_content = re.sub(r'\s+', ' ', joined_content).strip()
                    
                    if len(joined_content) > 300:
                        return joined_content
        except Exception as e:
            continue  # Try next pattern
    
    # If container approach failed, try direct paragraph extraction
    try:
        # Extract all paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
        
        # Filter out very short paragraphs (likely navigation or metadata)
        filtered_paragraphs = [p for p in paragraphs if len(p.strip()) > 100]
        
        if filtered_paragraphs and len(filtered_paragraphs) >= 3:
            # Clean HTML tags
            cleaned_paragraphs = [re.sub(r'<[^>]*>', ' ', p) for p in filtered_paragraphs]
            # Join with proper paragraph breaks
            joined_content = "\n\n".join(cleaned_paragraphs)
            # Clean whitespace
            joined_content = re.sub(r'\s+', ' ', joined_content).strip()
            
            if len(joined_content) > 300:
                return joined_content
    except Exception:
        pass
        
    return None

def extract_content_by_structure(html_content: str, url: str) -> Optional[str]:
    """Extract content by analyzing the document structure and density."""
    
    try:
        # Special handling for news sites we know
        domain = urllib.parse.urlparse(url).netloc.lower()
        
        # Site-specific extraction
        if 'nytimes.com' in domain:
            # NYT-specific pattern
            story_sections = re.findall(r'<section[^>]*name="articleBody"[^>]*>(.*?)</section>', html_content, re.DOTALL)
            if story_sections:
                paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', story_sections[0], re.DOTALL)
                if paragraphs:
                    return "\n\n".join([re.sub(r'<[^>]*>', ' ', p) for p in paragraphs])
        
        elif 'bbc.com' in domain or 'bbc.co.uk' in domain:
            # BBC-specific pattern 
            story_body = re.findall(r'<div[^>]*class="[^"]*story-body[^"]*"[^>]*>(.*?)</div>', html_content, re.DOTALL)
            if story_body:
                paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', story_body[0], re.DOTALL)
                if paragraphs:
                    return "\n\n".join([re.sub(r'<[^>]*>', ' ', p) for p in paragraphs])
        
        # Generic approach - find the "densest" part of the document
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
        
        # Score paragraphs by text density and proximity
        paragraph_scores = []
        for i, p in enumerate(paragraphs):
            text = re.sub(r'<[^>]*>', '', p).strip()
            # Score based on length and lack of boilerplate text
            if len(text) > 100 and not re.search(r'(copyright|all rights reserved|privacy policy|terms of use)', text, re.IGNORECASE):
                # Higher score for paragraphs near other content paragraphs (clustering)
                cluster_bonus = sum(1 for j in range(max(0, i-3), min(len(paragraphs), i+4)) 
                                   if len(re.sub(r'<[^>]*>', '', paragraphs[j]).strip()) > 80)
                paragraph_scores.append((i, len(text) + (cluster_bonus * 20)))
        
        # Sort by score
        paragraph_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top scoring paragraphs and sort by original order
        if paragraph_scores:
            top_indices = [ps[0] for ps in paragraph_scores[:min(15, len(paragraph_scores))]]
            top_indices.sort()  # Restore original order
            
            # Extract these paragraphs
            best_paragraphs = [re.sub(r'<[^>]*>', ' ', paragraphs[i]) for i in top_indices]
            content = "\n\n".join(best_paragraphs)
            
            if len(content) > 300:
                return content
    
    except Exception as e:
        pass  # Fall through to other methods
        
    return None

def extract_content_aggressive(html_content: str) -> Optional[str]:
    """Most aggressive content extraction as a last resort."""
    
    try:
        # Remove common non-content elements
        cleaned_html = html_content
        
        # Remove script, style, header, footer, nav, aside tags and their contents
        for tag in ['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']:
            cleaned_html = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract all text from remaining content
        text_blocks = []
        
        # First try paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', cleaned_html, re.DOTALL)
        for p in paragraphs:
            text = re.sub(r'<[^>]*>', ' ', p).strip()
            if len(text) > 80:
                text_blocks.append(text)
        
        # If not enough paragraphs, try divs
        if len("".join(text_blocks)) < 500:
            divs = re.findall(r'<div[^>]*>(.*?)</div>', cleaned_html, re.DOTALL)
            for div in divs:
                if len(div) > 200 and '<div' not in div.lower():  # Avoid nested divs
                    text = re.sub(r'<[^>]*>', ' ', div).strip()
                    if len(text) > 100:
                        text_blocks.append(text)
        
        # If still not enough, try other tags
        if len("".join(text_blocks)) < 500:
            for tag in ['section', 'article', 'main']:
                elements = re.findall(f'<{tag}[^>]*>(.*?)</{tag}>', cleaned_html, re.DOTALL)
                for el in elements:
                    text = re.sub(r'<[^>]*>', ' ', el).strip()
                    if len(text) > 150:
                        text_blocks.append(text)
        
        # Deduplicate and clean
        unique_blocks = []
        for block in text_blocks:
            clean_block = re.sub(r'\s+', ' ', block).strip()
            if clean_block and clean_block not in unique_blocks:
                unique_blocks.append(clean_block)
        
        # Sort by length (longest first) to prioritize main content
        unique_blocks.sort(key=len, reverse=True)
        
        # Take top blocks
        selected_blocks = unique_blocks[:min(10, len(unique_blocks))]
        
        # Check if we have enough content
        if selected_blocks and sum(len(b) for b in selected_blocks) > 300:
            return "\n\n".join(selected_blocks)
    
    except Exception:
        pass  # Last resort failed
        
    return None

def html_entity_decode(text: str) -> str:
    """Decode HTML entities in text."""
    
    # Handle common HTML entities
    entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&rdquo;': '"',
        '&ldquo;': '"',
        '&rsquo;': "'",
        '&lsquo;': "'",
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '…',
    }
    
    # Replace entities
    for entity, replacement in entities.items():
        text = text.replace(entity, replacement)
    
    # Handle numeric entities
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )

# Helper function to fetch news from RSS feeds
def fetch_articles_from_rss(query: NewsApiQuery = None) -> List[Dict[str, Any]]:
    """
    Fetch articles from RSS feeds.
    
    This function fetches articles from a list of reliable RSS feeds,
    formats them to match the NewsAPI format, and returns them.
    
    Args:
        query: Optional query parameters to filter articles
        
    Returns:
        List[Dict[str, Any]]: List of articles in NewsAPI format
    """
    # Select feeds to use (use all by default)
    feeds_to_use = FREE_NEWS_FEEDS
    
    # If query specifies domains, filter feeds
    if query and query.domains:
        domain_list = query.domains.split(',')
        feeds_to_use = [feed for feed in feeds_to_use if any(domain in feed for domain in domain_list)]
    
    # If no feeds match the domain filter, use all feeds
    if not feeds_to_use:
        feeds_to_use = FREE_NEWS_FEEDS
    
    # Randomize feed order to get variety
    random.shuffle(feeds_to_use)
    
    # Limit to max 5 feeds for performance
    feeds_to_use = feeds_to_use[:5]
    
    # Track all fetched articles
    all_articles = []
    
    # Process each feed
    for feed_url in feeds_to_use:
        try:
            # Parse the feed
            feed = feedparser.parse(feed_url)
            
            # Skip if feed parsing failed
            if not feed or not feed.entries:
                logger.warning(f"Failed to parse feed or no entries: {feed_url}")
                continue
                
            # Process entries
            for entry in feed.entries[:10]:  # Limit to 10 entries per feed
                # Skip entries that don't have required fields
                if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                    continue
                    
                # Extract publish date if available
                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_date = time.strftime('%Y-%m-%dT%H:%M:%SZ', entry.published_parsed)
                    except:
                        published_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                else:
                    published_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Extract description/summary
                description = ""
                if hasattr(entry, 'summary'):
                    description = entry.summary
                elif hasattr(entry, 'description'):
                    description = entry.description
                
                # Clean description of HTML tags
                description = re.sub(r'<[^>]+>', ' ', description)
                description = re.sub(r'\s+', ' ', description).strip()
                
                # Extract content if available
                content = ""
                if hasattr(entry, 'content'):
                    # Some feeds have content as a list of dictionaries
                    if isinstance(entry.content, list) and len(entry.content) > 0:
                        content = entry.content[0].value if hasattr(entry.content[0], 'value') else ""
                
                # Fallback to description if no content
                if not content:
                    content = description
                
                # Clean content of HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extract image URL if available
                image_url = None
                if hasattr(entry, 'media_content') and entry.media_content:
                    # Some feeds have media_content as a list of dictionaries
                    for media in entry.media_content:
                        if hasattr(media, 'url') and media.url:
                            image_url = media.url
                            break
                
                # Parse domain from feed URL
                domain = urllib.parse.urlparse(feed_url).netloc
                
                # Create article in NewsAPI format
                article = {
                    "source": {"id": domain, "name": feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else domain},
                    "author": entry.author if hasattr(entry, 'author') else None,
                    "title": entry.title,
                    "description": description[:300],  # Limit description length
                    "url": entry.link,
                    "urlToImage": image_url,
                    "publishedAt": published_date,
                    "content": content[:2000] if content else description[:500],  # Limit content length
                    "category": feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else "News"
                }
                
                # Apply query filters if provided
                if query:
                    # Filter by keyword if provided
                    if query.query and query.query.lower() not in article["title"].lower() and (
                            not article["description"] or query.query.lower() not in article["description"].lower()):
                        continue
                    
                    # Filter by language (simple heuristic)
                    if query.language and query.language != "en":
                        # Skip non-English articles for now
                        # In a real app, you'd use a language detection library
                        continue
                
                all_articles.append(article)
                
        except Exception as feed_err:
            logger.warning(f"Error processing feed {feed_url}: {str(feed_err)}")
            continue
    
    # If we got too few articles, try to get more from remaining feeds
    if len(all_articles) < 5 and len(feeds_to_use) < len(FREE_NEWS_FEEDS):
        remaining_feeds = [feed for feed in FREE_NEWS_FEEDS if feed not in feeds_to_use]
        for feed_url in remaining_feeds:
            try:
                feed = feedparser.parse(feed_url)
                if not feed or not feed.entries:
                    continue
                    
                for entry in feed.entries[:5]:
                    if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                        continue
                        
                    # Quick add with minimal processing
                    all_articles.append({
                        "source": {"id": urllib.parse.urlparse(feed_url).netloc, "name": feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else "News"},
                        "title": entry.title,
                        "description": entry.summary if hasattr(entry, 'summary') else "",
                        "url": entry.link,
                        "publishedAt": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "content": entry.summary if hasattr(entry, 'summary') else ""
                    })
                    
                    if len(all_articles) >= 10:
                        break
                        
                if len(all_articles) >= 10:
                    break
            except:
                continue
    
    # Log results
    logger.info(f"Fetched {len(all_articles)} articles from RSS feeds")
    
    # Ensure unique articles by title (deduplication)
    seen_titles = set()
    unique_articles = []
    
    for article in all_articles:
        if article["title"] not in seen_titles:
            seen_titles.add(article["title"])
            unique_articles.append(article)
    
    return unique_articles

# Helper functions for generating sample data
def generate_sample_news_articles(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample news articles for demonstration purposes."""
    
    categories = ["Business", "Technology", "Science", "Health", "Politics", "Entertainment", "Sports"]
    sources = ["The New York Times", "The Washington Post", "BBC News", "CNN", "Reuters", "Associated Press"]
    
    sample_titles = [
        "Global Leaders Announce New Climate Initiative at Summit",
        "Tech Giants Face Scrutiny in Antitrust Hearings",
        "Medical Researchers Report Breakthrough in Cancer Treatment",
        "Economic Recovery Shows Signs of Acceleration, Experts Say",
        "Political Tensions Rise Amid Congressional Deadlock",
        "New Study Reveals Impact of Social Media on Mental Health",
        "Sports League Announces Major Changes to Playoff Format",
        "Scientists Discover New Species in Remote Region",
        "Stock Market Reaches Record High Amid Strong Earnings",
        "Education Reform Bill Passes with Bipartisan Support",
        "Renewable Energy Investment Surges Globally",
        "Cultural Festival Celebrates Diversity and Tradition",
        "Transportation Department Unveils Infrastructure Plans",
        "Global Supply Chain Issues Continue to Impact Industries",
        "Housing Market Shows Signs of Cooling After Rapid Growth",
        "Space Agency Successfully Launches New Satellite",
        "Cybersecurity Threats Prompt New Government Guidelines",
        "Agricultural Innovation Addresses Food Security Concerns",
        "Consumer Spending Patterns Shift in Post-Pandemic Era",
        "International Relations Evolve Following Diplomatic Summit"
    ]
    
    sample_descriptions = [
        "World leaders from over 100 countries gathered to announce a groundbreaking climate initiative aimed at reducing carbon emissions.",
        "Major technology companies are facing intense scrutiny as antitrust hearings begin in Congress.",
        "A team of medical researchers has announced a significant breakthrough in the treatment of several types of cancer.",
        "Economic indicators suggest that the recovery is accelerating, with job growth and consumer spending on the rise.",
        "Political tensions continue to mount as Congress fails to reach agreement on key legislation.",
        "Researchers have published a comprehensive study on the complex relationship between social media usage and mental health outcomes.",
        "In a surprise announcement, the sports league has revealed major changes to its playoff structure for the upcoming season.",
        "A scientific expedition has documented several previously unknown species in a remote ecological zone.",
        "The stock market reached new heights today, driven by strong quarterly earnings reports across multiple sectors.",
        "After months of negotiation, a comprehensive education reform bill has passed with support from both political parties.",
        "Global investment in renewable energy solutions has reached unprecedented levels, according to a new industry report.",
        "The annual cultural festival brought together thousands to celebrate diverse traditions and promote cross-cultural understanding.",
        "Transportation officials have unveiled ambitious plans to revitalize aging infrastructure across the country.",
        "Manufacturing and retail industries continue to face challenges due to ongoing global supply chain disruptions.",
        "After years of rapid growth, the housing market is showing initial signs of moderation according to real estate analysts.",
        "The space agency has successfully deployed a new satellite designed to monitor climate patterns and environmental changes.",
        "In response to growing threats, government agencies have issued new cybersecurity guidelines for critical infrastructure.",
        "Innovative agricultural techniques are being developed to address growing concerns about global food security.",
        "New data indicates significant shifts in consumer spending patterns following the pandemic, with implications for retailers.",
        "Diplomatic relations are evolving following a high-level international summit addressing global challenges."
    ]
    
    articles = []
    # Real news sources for example URLs
    real_news_domains = [
        "bbc.com/news", 
        "apnews.com/article", 
        "reuters.com/world",
        "theguardian.com/us-news",
        "nytimes.com/section/world",
        "washingtonpost.com/world",
        "aljazeera.com/news",
        "nbcnews.com/news",
        "cnn.com/world",
        "bloomberg.com/news"
    ]
    
    for i in range(min(count, len(sample_titles))):
        current_time = datetime.now() - timedelta(hours=random.randint(1, 72))
        published_at = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        domain = random.choice(real_news_domains)
        
        # Generate realistic article ID for URL
        article_slug = sample_titles[i].lower().replace(" ", "-").replace(",", "").replace(".", "")
        article_slug = re.sub(r'[^a-zA-Z0-9-]', '', article_slug)
        article_id = f"{int(current_time.timestamp())}-{article_slug[:40]}"
        
        # Create more realistic content with paragraphs
        paragraphs = [
            sample_descriptions[i],
            f"The announcement comes as experts continue to debate the implications for {random.choice(['global policy', 'international relations', 'economic markets', 'public health', 'climate change'])}.",
            f"According to {random.choice(['analysts', 'officials', 'experts', 'researchers'])}, this development could have significant {random.choice(['long-term', 'short-term', 'immediate', 'future'])} implications.",
            f"Several {random.choice(['organizations', 'countries', 'institutions', 'agencies'])} have already responded with their own initiatives to address these concerns.",
            f"Critics argue that more needs to be done, while supporters praise this as a step in the right direction."
        ]
        content = "\n\n".join(paragraphs)
        
        # Create article with realistic attributes
        articles.append({
            "source": {"id": domain.split(".")[0], "name": random.choice(sources)},
            "author": f"{random.choice(['John', 'Sarah', 'Michael', 'Emma', 'David', 'Maria', 'James', 'Jennifer'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia'])}",
            "title": sample_titles[i],
            "description": sample_descriptions[i],
            "url": f"https://www.{domain}/{article_id}",
            "urlToImage": f"https://picsum.photos/id/{random.randint(1, 1000)}/800/600",
            "publishedAt": published_at,
            "content": content,
            "category": random.choice(categories)
        })
    
    return articles

def generate_sample_article(sl: SiliconLayer, article_id: str) -> ArticleResponse:
    """Generate a sample article with metrics for demonstration purposes."""
    
    # Get a random sample article
    sample_articles = generate_sample_news_articles(1)
    article = sample_articles[0]
    
    # Convert to article request
    article_request = ArticleRequest(
        title=article["title"],
        abstract=article["description"],
        content=article["content"],
        source=article["source"]["name"],
        published_date=article["publishedAt"],
        category=article["category"],
        article_id=article_id,
        url=article["url"],
        image_url=article["urlToImage"]
    )
    
    # Get predictions
    df = preprocess_article(article_request)
    predictions = sl.predict(df)
    
    if len(predictions) == 0:
        # Generate random metrics if prediction fails
        return ArticleResponse(
            article_id=article_id,
            title=article["title"],
            abstract=article["description"],
            content=article["content"],
            source=article["source"]["name"],
            published_date=article["publishedAt"],
            category=article["category"],
            url=article["url"],
            image_url=article["urlToImage"],
            political_influence=random.uniform(0.2, 0.8),
            rhetoric_intensity=random.uniform(0.2, 0.8),
            information_depth=random.uniform(0.2, 0.8),
            information_depth_category=random.choice(["Overview", "Analysis", "In-depth"])
        )
    
    # Return article with metrics
    return ArticleResponse(
        article_id=article_id,
        title=article["title"],
        abstract=article["description"],
        content=article["content"],
        source=article["source"]["name"],
        published_date=article["publishedAt"],
        category=article["category"],
        url=article["url"],
        image_url=article["urlToImage"],
        political_influence=float(predictions["political_influence"].iloc[0]),
        rhetoric_intensity=float(predictions["rhetoric_intensity"].iloc[0]),
        information_depth=float(predictions["information_depth"].iloc[0]),
        information_depth_category=predictions["information_depth_category"].iloc[0]
    )

def generate_sample_articles(sl: SiliconLayer, count: int) -> List[ArticleResponse]:
    """Generate a list of sample articles with metrics for demonstration purposes."""
    
    # Get multiple unique sample articles
    sample_news = generate_sample_news_articles(count)
    articles = []
    
    # Process each sample article
    for i, article in enumerate(sample_news):
        article_id = f"sample_{i}_{int(datetime.now().timestamp())}"
        
        # Create article request
        article_request = ArticleRequest(
            title=article["title"],
            abstract=article["description"],
            content=article["content"],
            source=article["source"]["name"],
            published_date=article["publishedAt"],
            category=article["category"],
            article_id=article_id,
            url=article["url"],
            image_url=article["urlToImage"]
        )
        
        # Get predictions
        df = preprocess_article(article_request)
        predictions = sl.predict(df)
        
        if len(predictions) == 0:
            # Generate random metrics if prediction fails
            article_response = ArticleResponse(
                article_id=article_id,
                title=article["title"],
                abstract=article["description"],
                content=article["content"],
                source=article["source"]["name"],
                published_date=article["publishedAt"],
                category=article["category"],
                url=article["url"],
                image_url=article["urlToImage"],
                political_influence=random.uniform(0.2, 0.8),
                rhetoric_intensity=random.uniform(0.2, 0.8),
                information_depth=random.uniform(0.2, 0.8),
                information_depth_category=random.choice(["Overview", "Analysis", "In-depth"])
            )
        else:
            # Create article with model predictions
            article_response = ArticleResponse(
                article_id=article_id,
                title=article["title"],
                abstract=article["description"],
                content=article["content"],
                source=article["source"]["name"],
                published_date=article["publishedAt"],
                category=article["category"],
                url=article["url"],
                image_url=article["urlToImage"],
                political_influence=float(predictions["political_influence"].iloc[0]),
                rhetoric_intensity=float(predictions["rhetoric_intensity"].iloc[0]),
                information_depth=float(predictions["information_depth"].iloc[0]),
                information_depth_category=predictions["information_depth_category"].iloc[0]
            )
        
        articles.append(article_response)
    
    return articles

def update_articles_cache(new_articles: List[ArticleResponse]):
    """Update the global articles cache with new articles."""
    global ARTICLES_CACHE
    
    # Add new articles to the beginning of the cache
    ARTICLES_CACHE = new_articles + ARTICLES_CACHE
    
    # Limit cache size to 100 articles
    if len(ARTICLES_CACHE) > 100:
        ARTICLES_CACHE = ARTICLES_CACHE[:100]

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )