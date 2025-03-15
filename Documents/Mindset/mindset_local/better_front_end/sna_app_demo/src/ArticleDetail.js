import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import config from './config';
import './ArticleDetail.css';

const ArticleDetail = () => {
  const { id } = useParams();
  const [article, setArticle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showMetricsExplanation, setShowMetricsExplanation] = useState(false);

  useEffect(() => {
    // Fetch article when component mounts or id changes
    fetchArticle();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);  // fetchArticle is intentionally excluded to prevent unnecessary re-renders

  const fetchArticle = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Add timeout to the request
      const response = await axios.get(`${config.API_URL}/article/${id}`, {
        timeout: config.DEFAULT_TIMEOUT,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      // Check if we got a valid response
      if (response.data && response.data.article_id) {
        setArticle(response.data);
        setLoading(false);
      } else {
        throw new Error('Invalid article data received');
      }
    } catch (err) {
      console.error('Error fetching article:', err);
      
      // Provide more specific error messages based on the error type
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server took too long to respond. Please try again later.');
      } else if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        if (err.response.status === 404) {
          setError('Article not found. It may have been removed or the ID is invalid.');
        } else {
          setError(`Server error (${err.response.status}). Please try again later.`);
        }
      } else if (err.request) {
        // The request was made but no response was received
        setError('No response from server. Please check your internet connection and try again.');
      } else {
        // Something happened in setting up the request that triggered an Error
        setError('Failed to fetch article. Please try again later.');
      }
      
      setLoading(false);
    }
  };

  // Function to format date to a readable format
  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  // Calculate the color for political influence (green to red)
  const getPoliticalColor = (value) => {
    const green = Math.round(52 * (1 - value));
    const red = Math.round(239 * value);
    return `rgb(${red}, ${green}, 73)`;
  };

  // Calculate the color for rhetoric intensity (blue to red)
  const getRhetoricColor = (value) => {
    const blue = Math.round(59 * (1 - value));
    const red = Math.round(239 * value);
    return `rgb(${red}, 73, ${blue})`;
  };

  // Get appropriate color class for information depth
  const getDepthColorClass = (category) => {
    if (!category) return 'depth-medium';
    
    const lowerCategory = category.toLowerCase();
    if (lowerCategory.includes('overview')) return 'depth-low';
    if (lowerCategory.includes('analysis')) return 'depth-medium';
    if (lowerCategory.includes('in-depth')) return 'depth-high';
    return 'depth-medium';
  };

  if (loading) {
    return <div className="loading">Loading article...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  if (!article) {
    return <div className="error">Article not found.</div>;
  }

  return (
    <div className="article-detail-container">
      <div className="article-header">
        <Link to="/" className="back-link">← Back to all articles</Link>
        <h1 className="article-title">{article.title}</h1>
        
        <div className="article-meta">
          {article.source && <span className="source">{article.source}</span>}
          {article.published_date && (
            <span className="date">{formatDate(article.published_date)}</span>
          )}
          {article.category && <span className="category">{article.category}</span>}
        </div>
      </div>

      {article.image_url && (
        <div className="article-image">
          <img src={article.image_url} alt={article.title} />
        </div>
      )}

      <div className="metrics-container">
        <h2 className="metrics-title">
          MINDSET Analysis
          <button 
            className="info-button" 
            onClick={() => setShowMetricsExplanation(!showMetricsExplanation)}
          >
            {showMetricsExplanation ? 'Hide Info' : 'What is this?'}
          </button>
        </h2>
        
        {showMetricsExplanation && (
          <div className="metrics-explanation">
            <p><strong>Political Influence:</strong> Measures the degree of political bias in the content, from neutral (green) to strongly biased (red).</p>
            <p><strong>Rhetoric Intensity:</strong> Analyzes the emotional charge and persuasive techniques used, from factual (blue) to emotionally charged (red).</p>
            <p><strong>Information Depth:</strong> Evaluates how comprehensive the information is, categorized as Overview, Analysis, or In-depth.</p>
          </div>
        )}
        
        <div className="metrics-display">
          <div className="metric">
            <div className="metric-header">
              <span className="metric-name">Political Influence</span>
              <span className="metric-value">{Math.round(article.political_influence * 100)}%</span>
            </div>
            <div className="meter">
              <div 
                className="meter-fill" 
                style={{ 
                  width: `${article.political_influence * 100}%`, 
                  backgroundColor: getPoliticalColor(article.political_influence) 
                }}
              ></div>
            </div>
          </div>
          
          <div className="metric">
            <div className="metric-header">
              <span className="metric-name">Rhetoric Intensity</span>
              <span className="metric-value">{Math.round(article.rhetoric_intensity * 100)}%</span>
            </div>
            <div className="meter">
              <div 
                className="meter-fill" 
                style={{ 
                  width: `${article.rhetoric_intensity * 100}%`, 
                  backgroundColor: getRhetoricColor(article.rhetoric_intensity) 
                }}
              ></div>
            </div>
          </div>
          
          <div className="metric">
            <div className="metric-header">
              <span className="metric-name">Information Depth</span>
              <span className={`depth-badge ${getDepthColorClass(article.information_depth_category)}`}>
                {article.information_depth_category}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="article-content">
        <div className="article-abstract">
          <p>{article.abstract}</p>
        </div>
        
        <div className="article-body">
          {article.content ? (
            <p>{article.content}</p>
          ) : (
            <p>No content available for this article.</p>
          )}
        </div>
        
        {article.url && (
          <div className="article-source-link">
            <a href={article.url} target="_blank" rel="noopener noreferrer">
              Read original article
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default ArticleDetail;