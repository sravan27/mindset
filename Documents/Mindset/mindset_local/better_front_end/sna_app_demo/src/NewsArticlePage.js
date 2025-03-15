import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './NewsArticlePage.css';

// Note: axios and API_URL removed as they are not used in this component

const CircularProgress = ({ percentage, size, color, strokeWidth = 3, showText = false }) => {
  // Animate from 0 to target percentage
  const [currentPercentage, setCurrentPercentage] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => setCurrentPercentage(percentage), 100);
    return () => clearTimeout(timer);
  }, [percentage]);

  const adjustedRadius = (size - strokeWidth * 2 - 4) / 2;
  const circumference = 2 * Math.PI * adjustedRadius;
  const offset = circumference * (1 - currentPercentage / 100);

  return (
    <svg width={size} height={size} className="progress-svg">
      <circle
        className="progress-bg"
        stroke="#e6e6e6"
        strokeWidth={strokeWidth}
        fill="none"
        cx={size / 2}
        cy={size / 2}
        r={adjustedRadius}
      />
      <circle
        className="progress-bar"
        stroke={color}
        strokeWidth={strokeWidth}
        fill="none"
        cx={size / 2}
        cy={size / 2}
        r={adjustedRadius}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
      />
      {showText && (
        <text
          x="50%"
          y="50%"
          dominantBaseline="central"
          textAnchor="middle"
          className="progress-text"
          transform={`rotate(90, ${size / 2}, ${size / 2})`}
        >
          {Math.round(percentage)}%
        </text>
      )}
    </svg>
  );
};

const NewsArticlePage = ({ articles = [], loading = false, error = null, refreshArticles = () => {} }) => {
  const [modalOpen, setModalOpen] = useState(false);
  const [explainOpen, setExplainOpen] = useState(false);
  const [selectedFeedback, setSelectedFeedback] = useState(null);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');

  const openModal = (article) => {
    setSelectedArticle(article);
    setModalOpen(true);
  };
  
  const closeModal = (e) => {
    if (e) e.stopPropagation();
    setModalOpen(false);
    setSelectedArticle(null);
  };

  const toggleExplain = (e) => {
    e.stopPropagation();
    setExplainOpen((prev) => !prev);
  };

  const handleFeedback = (feedback) => {
    setSelectedFeedback(feedback);
  };

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleCategoryFilter = (category) => {
    setCategoryFilter(category === categoryFilter ? '' : category);
  };

  // Filter articles based on search term and category
  const filteredArticles = articles.filter(article => {
    const matchesSearch = searchTerm === '' || 
      article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      article.abstract.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesCategory = categoryFilter === '' || 
      article.category === categoryFilter;
    
    return matchesSearch && matchesCategory;
  });

  // Extract unique categories from articles
  const categories = Array.from(new Set(articles.map(article => article.category))).filter(Boolean);

  // Convert API article metrics to percentages for the UI
  const getPercentages = (article) => {
    return {
      politicalBias: article.political_influence * 100,
      rhetoricIntensity: article.rhetoric_intensity * 100,
      informationDepth: article.information_depth * 100,
      depthCategory: article.information_depth_category
    };
  };

  return (
    <div className="news-page-container">
      <div className="search-bar-container">
        <input 
          type="text" 
          placeholder="Search articles..." 
          value={searchTerm}
          onChange={handleSearch}
          className="search-input"
        />
        
        <button 
          className="refresh-button" 
          onClick={refreshArticles}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="category-filters">
        <span 
          className={`category-filter ${categoryFilter === '' ? 'active' : ''}`}
          onClick={() => handleCategoryFilter('')}
        >
          All
        </span>
        
        {categories.map(category => (
          <span 
            key={category} 
            className={`category-filter ${categoryFilter === category ? 'active' : ''}`}
            onClick={() => handleCategoryFilter(category)}
          >
            {category}
          </span>
        ))}
      </div>
      
      {loading ? (
        <div className="loading-container">
          Loading articles...
        </div>
      ) : filteredArticles.length === 0 ? (
        <div className="no-results">
          No articles found. Try adjusting your search or filters.
        </div>
      ) : (
        <div className="articles-grid">
          {filteredArticles.map(article => {
            const metrics = getPercentages(article);
            
            return (
              <div key={article.article_id} className="article-card">
                {article.image_url && (
                  <div className="article-image">
                    <img src={article.image_url} alt={article.title} />
                    {article.category && (
                      <span className="article-category">{article.category}</span>
                    )}
                  </div>
                )}
                
                <div className="article-content">
                  <h3 className="article-title">
                    <Link to={`/article/${article.article_id}`}>
                      {article.title}
                    </Link>
                  </h3>
                  
                  <p className="article-abstract">
                    {article.abstract.length > 150 
                      ? `${article.abstract.substring(0, 150)}...` 
                      : article.abstract}
                  </p>
                  
                  <div className="article-meta">
                    {article.source && <span className="source">{article.source}</span>}
                    {article.published_date && (
                      <span className="date">
                        {new Date(article.published_date).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                  
                  <div className="article-metrics" onClick={() => openModal(article)}>
                    <div className="metric-item">
                      <span className="metric-label">Political</span>
                      <div className="metric-bar">
                        <div 
                          className="metric-fill"
                          style={{
                            width: `${metrics.politicalBias}%`,
                            backgroundColor: `rgb(${Math.round(239 * metrics.politicalBias/100)}, ${Math.round(52 * (1 - metrics.politicalBias/100))}, 73)`
                          }}
                        ></div>
                      </div>
                      <span className="metric-value">{Math.round(metrics.politicalBias)}%</span>
                    </div>
                    
                    <div className="metric-item">
                      <span className="metric-label">Rhetoric</span>
                      <div className="metric-bar">
                        <div 
                          className="metric-fill"
                          style={{
                            width: `${metrics.rhetoricIntensity}%`,
                            backgroundColor: `rgb(${Math.round(239 * metrics.rhetoricIntensity/100)}, 73, ${Math.round(59 * (1 - metrics.rhetoricIntensity/100))})`
                          }}
                        ></div>
                      </div>
                      <span className="metric-value">{Math.round(metrics.rhetoricIntensity)}%</span>
                    </div>
                    
                    <div className="metric-depth">
                      <span className="metric-label">Depth</span>
                      <span className={`depth-badge depth-${metrics.depthCategory.toLowerCase().replace('-', '')}`}>
                        {metrics.depthCategory}
                      </span>
                    </div>
                  </div>
                  
                  <Link to={`/article/${article.article_id}`} className="read-more">
                    Read Full Article
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Modal Overlay */}
      {modalOpen && selectedArticle && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={closeModal}>×</button>
            <h3 className="modal-title">{selectedArticle.title}</h3>
            
            <div className="modal-metrics">
              <div className="modal-metric">
                <h4>Political Influence</h4>
                <CircularProgress
                  percentage={getPercentages(selectedArticle).politicalBias}
                  size={120}
                  strokeWidth={10}
                  color={`rgb(${Math.round(239 * selectedArticle.political_influence)}, ${Math.round(52 * (1 - selectedArticle.political_influence))}, 73)`}
                  showText
                />
                <p className="metric-description">
                  Measures the article's political bias from neutral to strongly biased
                </p>
              </div>
              
              <div className="modal-metric">
                <h4>Rhetoric Intensity</h4>
                <CircularProgress
                  percentage={getPercentages(selectedArticle).rhetoricIntensity}
                  size={120}
                  strokeWidth={10}
                  color={`rgb(${Math.round(239 * selectedArticle.rhetoric_intensity)}, 73, ${Math.round(59 * (1 - selectedArticle.rhetoric_intensity))})`}
                  showText
                />
                <p className="metric-description">
                  Measures the emotional intensity and persuasive language
                </p>
              </div>
              
              <div className="modal-metric">
                <h4>Information Depth</h4>
                <CircularProgress
                  percentage={getPercentages(selectedArticle).informationDepth}
                  size={120}
                  strokeWidth={10}
                  color="#4299e1"
                  showText
                />
                <p className="metric-description">
                  Category: <span className={`depth-badge depth-${selectedArticle.information_depth_category.toLowerCase().replace('-', '')}`}>
                    {selectedArticle.information_depth_category}
                  </span>
                </p>
              </div>
            </div>
            
            <button className="explain-button" onClick={toggleExplain}>
              {explainOpen ? 'Hide Explanation' : 'What Do These Metrics Mean?'}
            </button>
            
            {explainOpen && (
              <div className="explain-content">
                <p>
                  <strong>Political Influence:</strong> Measures the degree of political bias in the content, from neutral (green) to strongly biased (red).
                </p>
                <p>
                  <strong>Rhetoric Intensity:</strong> Analyzes the emotional charge and persuasive techniques used, from factual (blue) to emotionally charged (red).
                </p>
                <p>
                  <strong>Information Depth:</strong> Evaluates how comprehensive the information is, categorized as Overview, Analysis, or In-depth.
                </p>
              </div>
            )}

            <div className="feedback-section">
              <p className="feedback-prompt">Do these metrics accurately reflect this article?</p>
              <div className="feedback-buttons">
                <button
                  className={`feedback-button ${selectedFeedback === "up" ? "active" : ""}`}
                  onClick={() => handleFeedback("up")}
                >
                  👍 Yes
                </button>
                <button
                  className={`feedback-button ${selectedFeedback === "down" ? "active" : ""}`}
                  onClick={() => handleFeedback("down")}
                >
                  👎 No
                </button>
              </div>
              
              {selectedFeedback && (
                <div className="feedback-thanks">
                  Thank you for your feedback!
                </div>
              )}
            </div>
            
            <Link to={`/article/${selectedArticle.article_id}`} className="read-full-button">
              Read Full Article
            </Link>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewsArticlePage;