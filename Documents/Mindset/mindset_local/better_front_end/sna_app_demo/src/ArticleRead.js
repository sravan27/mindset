import React, { useState } from 'react';
import './ArticleRead.css';
import axios from 'axios';
import config from './config';

const ArticleRead = () => {
  const [url, setUrl] = useState('');
  const [article, setArticle] = useState(null);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Function to fetch and parse the article using our backend
  const scrapeArticle = async (articleUrl) => {
    setError('');
    
    // Validate the URL format first
    try {
      new URL(articleUrl); // This will throw if URL is invalid
    } catch (urlError) {
      throw new Error('Invalid URL format. Please enter a complete URL (including https://).');
    }
    
    // Set a specific user message while fetching
    setError('Fetching article content... This may take a few moments.');
    
    try {
      // Always use our backend's fetch article endpoint
      const backendFetchUrl = `${config.API_URL}/fetch-article?url=${encodeURIComponent(articleUrl)}`;
      
      // Create request with longer timeout since some news sites load slowly
      const backendResponse = await axios.get(backendFetchUrl, {
        timeout: 30000, // 30 seconds timeout for news sites that load slowly
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      // Clear the "fetching" message
      setError('');
      
      if (backendResponse.data && backendResponse.data.content) {
        // Success - we got content from the backend
        return {
          title: backendResponse.data.title || "No title available",
          image: backendResponse.data.image_url || "",
          content: backendResponse.data.content,
          description: backendResponse.data.description || "",
          source: backendResponse.data.source || new URL(articleUrl).hostname,
          url: articleUrl
        };
      } else {
        // The backend didn't return an error, but didn't give us content either
        throw new Error('Could not extract content from the article. The site may be blocking content extraction.');
      }
    } catch (error) {
      // Clear the "fetching" message
      setError('');
      
      console.error('Article fetch error:', error);
      
      // Provide more helpful error messages based on the error type
      if (error.response) {
        // The request was made and the server responded with an error status
        if (error.response.status === 404) {
          throw new Error('The article URL could not be found (404 error). Please check the URL and try again.');
        } else if (error.response.status === 403) {
          throw new Error('Access to this content is forbidden (403 error). The site may be blocking our requests.');
        } else {
          throw new Error(`Server error (${error.response.status}): ${error.response.data.detail || 'Could not fetch article'}`);
        }
      } else if (error.request) {
        // The request was made but no response was received
        throw new Error('No response received from the server. Please check your internet connection and try again.');
      } else {
        // Something else happened in setting up the request
        throw new Error(`Failed to fetch article: ${error.message}`);
      }
    }
  };

  // Analyze the article with our mindset API
  const analyzeArticle = async (articleData) => {
    setAnalyzing(true);
    setError(''); // Clear any previous error
    
    try {
      // Ensure we have sufficient content to analyze
      if (!articleData.content || articleData.content.length < 100) {
        throw new Error('Insufficient content to analyze. The article text is too short.');
      }
      
      // Prepare a good abstract - either use the description or first part of content
      const abstract = articleData.description && articleData.description.length > 50 
        ? articleData.description 
        : articleData.content.substring(0, 250);
      
      // Generate a unique ID for the article
      const article_id = `article_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
      
      // Display a status message during analysis
      setError('Analyzing article content...');
      
      // Make the analysis request with a longer timeout
      const response = await axios.post(`${config.API_URL}/analyze`, {
        title: articleData.title,
        abstract: abstract,
        content: articleData.content,
        source: articleData.source,
        article_id: article_id,
        url: articleData.url
      }, {
        timeout: 40000, // 40 seconds timeout for model analysis
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      // Clear status message
      setError('');
      
      // Validate the response
      if (!response.data || 
          typeof response.data.political_influence !== 'number' || 
          typeof response.data.rhetoric_intensity !== 'number' || 
          typeof response.data.information_depth !== 'number') {
        throw new Error('Invalid response from analysis API. Missing metrics data.');
      }
      
      // Set the metrics in state
      setMetrics({
        political_influence: response.data.political_influence,
        rhetoric_intensity: response.data.rhetoric_intensity,
        information_depth: response.data.information_depth,
        information_depth_category: response.data.information_depth_category || 'Analysis'
      });
      
      setAnalyzing(false);
    } catch (err) {
      console.error('Error analyzing article:', err);
      
      // Provide detailed error messages based on the error type
      if (err.response) {
        // The request was made and the server responded with an error status
        if (err.response.status === 500) {
          setError('Server error during analysis. The article might be too complex or in an unsupported format.');
        } else {
          setError(`Analysis failed: ${err.response.data?.detail || 'Server error'}`);
        }
      } else if (err.request) {
        // The request was made but no response was received
        setError('Analysis request timed out. The server might be busy or the article too long.');
      } else {
        // Something happened in setting up the request or processing the response
        setError(`Analysis failed: ${err.message || 'Unknown error'}`);
      }
      
      setAnalyzing(false);
    }
  };

  const handleScrape = async () => {
    setError('');
    setArticle(null);
    setMetrics(null);
    setLoading(true);
    
    try {
      const scraped = await scrapeArticle(url);
      setArticle(scraped);
      setLoading(false);
      
      // After successful scraping, analyze the article
      await analyzeArticle(scraped);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Function to display metrics with proper formatting
  const displayMetrics = () => {
    if (!metrics) return null;
    
    return (
      <div className="article-metrics">
        <h3>MINDSET Analysis</h3>
        <div className="metrics-container">
          <div className="metric-item">
            <div className="metric-label">Political Influence</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${metrics.political_influence * 100}%`,
                  backgroundColor: `rgb(${Math.round(239 * metrics.political_influence)}, ${Math.round(52 * (1 - metrics.political_influence))}, 73)`
                }}
              ></div>
            </div>
            <div className="metric-value">{Math.round(metrics.political_influence * 100)}%</div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Rhetoric Intensity</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${metrics.rhetoric_intensity * 100}%`,
                  backgroundColor: `rgb(${Math.round(239 * metrics.rhetoric_intensity)}, 73, ${Math.round(59 * (1 - metrics.rhetoric_intensity))})`
                }}
              ></div>
            </div>
            <div className="metric-value">{Math.round(metrics.rhetoric_intensity * 100)}%</div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Information Depth</div>
            <div className="metric-value depth-category">
              <span className={`depth-badge depth-${metrics.information_depth_category.toLowerCase().replace('-', '')}`}>
                {metrics.information_depth_category}
              </span>
              <span className="depth-percent">({Math.round(metrics.information_depth * 100)}%)</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="article-read">
      <h2>Read and Analyze a News Article</h2>
      <div className="read-form">
        <input
          type="text"
          placeholder="Enter article URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading || analyzing}
        />
        <button 
          onClick={handleScrape}
          disabled={loading || analyzing || !url.trim()}
        >
          {loading ? 'Loading...' : analyzing ? 'Analyzing...' : 'Read & Analyze'}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      
      {article && (
        <div className="scraped-article">
          <h3>{article.title}</h3>
          {article.image && (
            <div className="article-image">
              <img src={article.image} alt={article.title} />
            </div>
          )}
          
          {/* Display metrics if available */}
          {metrics && displayMetrics()}
          
          <div className="article-content">
            <p>{article.content}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ArticleRead;
